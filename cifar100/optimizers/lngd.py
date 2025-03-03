from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, trainable_modules, momentum_step


def try_contiguous(x):
    return x if x.is_contiguous() else x.contiguous()


class LNGD(Optimizer):
    """
    Layer-wise Natural Gradient Descent (LNGD) optimizer.
    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Base learning rate.
        momentum (float): Momentum factor.
        stat_decay (float): Decay rate for exponential moving averages.
        damping (float): Base damping constant.
        weight_decay (float): Weight decay factor.
        Tcov (int): Interval (in steps) for updating running statistics.
        Tinv (int): Interval (in steps) for updating the inverses.
        mu (float): Small constant in adaptive learning rate denominator.
        nu1 (float): Minimum damping bound.
        nu2 (float): Maximum damping bound.
    """

    def __init__(
            self,
            params,
            lr=0.1,
            momentum=0.9,
            stat_decay=0.95,
            damping=1.0,
            weight_decay=5e-4,
            Tcov=5,
            Tinv=50,
            mu=1e-3,
            nu1=1e-5,
            nu2=1e-2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if stat_decay < 0.0:
            raise ValueError(f"Invalid stat_decay value: {stat_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr,
                        momentum=momentum,
                        stat_decay=stat_decay,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._model = None
        self.damping = damping
        self.Tcov = Tcov
        self.Tinv = Tinv
        self.mu = mu
        self.nu1 = nu1
        self.nu2 = nu2
        self._step = 0
        self.emastep = 0

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        # Register forward and backward hooks to capture statistics.
        self.layer_map = build_layer_map(model,
                                         fwd_hook_fn=self._capture_activation,
                                         bwd_hook_fn=self._capture_backprop)

    def _configure(self, train_loader, net, device):
        r"""
        (Optional) Pre-compute statistics for the first layer using a subset of data.
        For LNGD, we pre-compute the first layer’s input covariance inverse.
        """
        n_batches = len(train_loader)
        cov_mat = None

        # Use the first trainable module as the first layer.
        _, first_layer = next(trainable_modules(net))

        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(device, non_blocking=True)
                actv = extract_patches(images, first_layer.kernel_size,
                                       first_layer.stride, first_layer.padding,
                                       depthwise=False)
                if first_layer.bias is not None:
                    ones = actv.new_ones((actv.shape[0], 1))
                    actv = torch.cat([actv, ones], dim=1)
                A = torch.matmul(actv.t(), actv) / actv.size(0)
                cov_mat = A if cov_mat is None else cov_mat.add_(A)
            cov_mat /= n_batches

        self.first_layer = first_layer
        eye_matrix = torch.eye(cov_mat.size(0), device=device, dtype=cov_mat.dtype)
        self.input_cov_inv = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat + self.damping * eye_matrix))
        self.model = net
        self.layer_map[first_layer]['fwd_hook'].remove()

    def _capture_activation(
            self,
            module: nn.Module,
            forward_input: List[torch.Tensor],
            _forward_output: torch.Tensor
    ):
        r"""
        Forward hook: capture activations aₗ₋₁.
        Computes the activation covariance and the average squared norm:
          A_est = (aᵀa)/B  and  a_norm_sq = mean(||a||²).
        """
        if not module.training or not torch.is_grad_enabled():
            return
        if (self._step % self.Tcov) != 0:
            return

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        actv = forward_input[0].detach().clone()
        # For Conv2d layers, extract patches and flatten per sample.
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
            actv = actv.view(actv.size(0), -1)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:
                actv = actv.view(-1, actv.size(-1))
        # (Optionally, do not append bias for computing covariance.)
        B = actv.size(0)
        A_est = torch.matmul(actv.t(), actv) / B  # [d, d]
        a_norm_sq = (actv.pow(2).sum(dim=1)).mean()  # scalar

        state = self.state[module]
        if 'A' not in state:
            state['A'] = A_est.clone()
            state['a_norm_sq'] = a_norm_sq.clone()
        else:
            state['A'].mul_(stat_decay).add_(A_est, alpha=1 - stat_decay)
            state['a_norm_sq'] = state['a_norm_sq'] * stat_decay + (1 - stat_decay) * a_norm_sq

    def _capture_backprop(
            self,
            module: nn.Module,
            _grad_input: torch.Tensor,
            grad_output: torch.Tensor
    ):
        r"""
        Backward hook: capture the output gradients gₗ.
        Computes:
          g_norm_sq = mean(∥g∥²)   and   g_square = mean(g²)  (elementwise).
        """
        if (self._step % self.Tcov) != 0:
            return
        group = self.param_groups[0]
        stat_decay = group['stat_decay']

        g = grad_output[0].detach().clone()
        if isinstance(module, nn.Conv2d):
            g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(g.size(0), -1)  # [B, m]
        B = g.size(0)
        g_norm_sq = (g.pow(2).sum(dim=1)).mean()  # scalar: E[∥g∥²]
        g_square = g.pow(2).mean(dim=0)  # vector: E[g²] for each output unit

        state = self.state[module]
        if 'g_norm_sq' not in state:
            state['g_norm_sq'] = g_norm_sq.clone()
            state['g_square'] = g_square.clone()
        else:
            state['g_norm_sq'] = state['g_norm_sq'] * stat_decay + (1 - stat_decay) * g_norm_sq
            state['g_square'] = state['g_square'] * stat_decay + (1 - stat_decay) * g_square

    @torch.no_grad()
    def step(self, closure=None):
        r"""
        Performs a single LNGD update step.

        For each layer (except the first, which uses a pre-computed input covariance inverse),
        the algorithm:
          1. Retrieves the bias-corrected estimates of activation covariance (A) and gradient statistics.
          2. Computes the factors:
                Φ = (A) · (g_norm_sq)   and   Ψ = diag(g_square) / (g_norm_sq).
          3. Applies damping to obtain Φ̂ and Ψ̂, then computes their inverses.
          4. Preconditions the reshaped gradient G as:
                natural_grad = Ψ̂⁻¹ · G · Φ̂⁻¹.
          5. Computes an adaptive layer-wise learning rate and applies momentum.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        b_updated = False
        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = self.damping
        lr = group['lr']

        if self._step % self.Tinv == 0:
            b_updated = True
            self.emastep += 1

        for layer in self.layer_map:
            if not (isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None):
                continue
            state = self.state[layer]
            grad_mat = reshape_grad(layer)  # shape: [m, d] (including bias column if present)

            if layer == self.first_layer:
                # Use the precomputed input covariance inverse for the first layer.
                natural_grad = grad_mat @ self.input_cov_inv.to(grad_mat.dtype)
            else:
                if b_updated:
                    bias_correction = 1.0 - (stat_decay ** self.emastep)
                    # Retrieve bias-corrected statistics.
                    A_est = state['A'].div(bias_correction)
                    g_norm_sq = state['g_norm_sq'].div(bias_correction)
                    g_square = state['g_square'].div(bias_correction)

                    # Compute factor Φ = A_est * g_norm_sq. (Dimension: [d, d])
                    Phi = A_est * g_norm_sq
                    # Compute factor Ψ = diag(g_square) / (g_norm_sq). (Dimension: [m, m])
                    # Here, g_square is a vector of length m.
                    eps = 1e-8
                    psi_diag = g_square / (g_norm_sq + eps)
                    # Form Ψ as a diagonal matrix.
                    Psi = torch.diag(psi_diag)

                    # Optionally, one might also damp Φ and Ψ.
                    I_d = torch.eye(Phi.size(0), device=Phi.device, dtype=Phi.dtype)
                    I_m = torch.eye(Psi.size(0), device=Psi.device, dtype=Psi.dtype)
                    Phi_hat = Phi + damping * I_d
                    Psi_hat = Psi + damping * I_m

                    # Compute inverses.
                    Phi_inv = torch.cholesky_inverse(torch.linalg.cholesky(Phi_hat))
                    # For the diagonal Ψ_hat, inversion is element-wise.
                    psi_inv_diag = 1.0 / (psi_diag + damping)
                    Psi_inv = torch.diag(psi_inv_diag)

                natural_grad = Psi_inv @ grad_mat @ Phi_inv

            # Compute an inner product between the natural gradient and the original gradient.
            dot_val = torch.sum(natural_grad * grad_mat)
            # Adaptive layer-wise learning rate: αₗ = dot_val / (dot_val + μ)
            adaptive_lr = dot_val / (dot_val + self.mu) if (dot_val + self.mu) != 0 else 1.0

            # Final update: scaled by the base learning rate and the adaptive factor.
            update_direction = -lr * adaptive_lr * natural_grad

            # If the layer has a bias, split the update accordingly.
            if layer.bias is not None:
                weight_update = update_direction[:, :-1]
                bias_update = update_direction[:, -1:]
                layer.weight.grad.data.copy_(weight_update.view_as(layer.weight))
                layer.bias.grad.data.copy_(bias_update.view_as(layer.bias))
            else:
                layer.weight.grad.data.copy_(update_direction.view_as(layer.weight.grad))

        momentum_step(self)
        self._step += 1
        return loss
