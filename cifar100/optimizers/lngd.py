from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, momentum_step


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
        self.layer_map = build_layer_map(model,
                                         fwd_hook_fn=self._capture_activation,
                                         bwd_hook_fn=self._capture_backprop)

    def _capture_activation(
            self,
            module: nn.Module,
            forward_input: List[torch.Tensor],
            _forward_output: torch.Tensor
    ):
        if not module.training or not torch.is_grad_enabled():
            return
        if (self._step % self.Tcov) != 0:
            return

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        actv = forward_input[0].detach().clone()

        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:
                actv = actv.view(-1, actv.size(-1))

        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1) # [B, d_in]

        if isinstance(module, nn.Conv2d):
            # Compute original batch size and number of patches per image
            B = forward_input[0].size(0)
            P = actv.size(0) // B
            # Reshape to [B, P, d_in]
            actv = actv.view(B, P, actv.size(-1))
            actv = actv.mean(dim=1)
            a_norm_sq = actv.pow(2).sum(dim=(1, 2))  # [B, ]
        else:
            a_norm_sq = actv.pow(2).sum(dim=1)  # [B, ]

        state = self.state[module]
        if 'actv' not in state:
            state['actv'] = torch.zeros_like(actv, device=actv.device)
            state['a_norm_sq'] = torch.zeros_like(a_norm_sq, device=a_norm_sq.device)
        state['actv'].mul_(stat_decay).add_(actv, alpha=1 - stat_decay)
        state['a_norm_sq'].mul_(stat_decay).add_(a_norm_sq, alpha=1 - stat_decay)

    def _capture_backprop(
            self,
            module: nn.Module,
            _grad_input: torch.Tensor,
            grad_output: torch.Tensor
    ):
        if (self._step % self.Tcov) != 0:
            return
        group = self.param_groups[0]
        stat_decay = group['stat_decay']

        g = grad_output[0].detach().clone()
        if isinstance(module, nn.Conv2d):
            g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1)) # [B, d_out]

        if isinstance(module, nn.Conv2d):
            B = grad_output[0].size(0)
            P = g.size(0) // B
            # Reshape to [B, H x W, d_out]
            g = g.view(B, P, g.size(-1))
            g_diag = g.pow(2).mean(dim=1)
            g_norm_sq = g.pow(2).sum(dim=(1,2))  # [B, ]
        else:
            g_diag = g.pow(2) # [B, d_out]
            g_norm_sq = g.pow(2).sum(dim=1)  # [B, ]

        state = self.state[module]
        if 'g_diag' not in state:
            state['g_diag'] = torch.zeros_like(g_diag, device=g_diag.device)
            state['g_norm_sq'] = torch.zeros_like(g_norm_sq, device=g_norm_sq.device)
        state['g_diag'].mul_(stat_decay).add_(g_diag, alpha=1 - stat_decay)
        state['g_norm_sq'].mul_(stat_decay).add_(g_norm_sq, alpha=1 - stat_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        b_updated = False
        group = self.param_groups[0]
        stat_decay = group['stat_decay']

        if self._step % self.Tinv == 0:
            b_updated = True
            self.emastep += 1

        for layer in self.layer_map:
            #layer_name = self.layer_map[layer]['name']
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)  # shape: [m, d] (including bias column if present)

                if b_updated:
                    bias_correction = 1.0 - (stat_decay ** self.emastep)
                    actv = state['actv'].div(bias_correction)
                    g_diag = state['g_diag'].div(bias_correction) # [B, d_out]
                    a_norm_sq = state['a_norm_sq'].div(bias_correction) # [B, ]
                    g_norm_sq = state['g_norm_sq'].div(bias_correction) # [B, ]

                    #if isinstance(layer, nn.Conv2d):
                    #    cov = torch.einsum('bij,bik->bjk', actv, actv)
                    #else:
                    #    cov = torch.einsum('bi,bj->bij', actv, actv)
                    cov = torch.einsum('bi,bj->bij', actv, actv)

                    phi = (cov * g_norm_sq.view(-1, 1, 1)).mean(dim=0)
                    psi = (g_diag * a_norm_sq.view(-1, 1)).mean(dim=0) / (a_norm_sq * g_norm_sq).mean(dim=0)

                    #damping_phi = (torch.trace(phi) / grad_mat.view(-1).size(0)).clamp(self.nu1, self.nu2)
                    damping_phi = (torch.trace(phi) / phi.size(0)).clamp(self.nu1, self.nu2)
                    #damping_psi = (torch.sum(psi) / grad_mat.view(-1).size(0)).clamp(self.nu1, self.nu2)
                    damping_psi = (torch.sum(psi) / psi.size(0)).clamp(self.nu1, self.nu2)

                    phi.add_(damping_phi)
                    psi.add_(damping_psi).reciprocal_()

                    state['A_inv'] = torch.linalg.inv(phi)
                    state['G_inv'] = torch.eye(psi.size(0), device=psi.device)
                    state['G_inv'].diagonal().copy_(psi)

                A_inv = state['A_inv']
                G_inv = state['G_inv']
                v = G_inv @ grad_mat @ A_inv

                # Adaptive layer-wise learning rate: dot_val / (dot_val ** 2 + mu)
                dot_val = torch.dot(v.view(-1), grad_mat.view(-1))
                adaptive_lr = dot_val / (dot_val ** 2 + self.mu)
                v_alr = adaptive_lr * v

                if layer.bias is not None:
                    v_alr = [v_alr[:, :-1], v_alr[:, -1:]]
                    layer.weight.grad.data.copy_(v_alr[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v_alr[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v_alr.view_as(layer.weight.grad))

        momentum_step(self)
        self._step += 1

        return loss
