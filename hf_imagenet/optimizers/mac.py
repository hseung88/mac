from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, trainable_modules, momentum_step


class MAC(Optimizer):
    def __init__(
            self,
            params,
            lr=0.1,
            momentum=0.9,
            stat_decay=0.99,
            damping=1.0,
            weight_decay=5e-4,
            Tcov=1,
            Tinv=5,
            vit_mode=True,  # Indicates ViT architecture; enables special handling
            cls_token_weight=0.5,  # Weight for the class token vs. patch tokens when aggregating
            use_nonlinear_transform=True  # If True, applies a tanh transform to the mean activation
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
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
        self._step = 0
        self.emastep = 0

        # ViT-specific settings
        self.vit_mode = vit_mode
        self.cls_token_weight = cls_token_weight
        self.use_nonlinear_transform = use_nonlinear_transform
        # When in ViT mode, you might prefer using the raw (pre‑LayerNorm) activations.
        self.use_pre_ln_stats = vit_mode

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model, fwd_hook_fn=self._capture_activation)

    def _configure(self, train_loader, net, device):
        n_batches = len(train_loader)
        cov_mat = None

        # For ViTs, if available, use the patch embedding projection
        if hasattr(net, 'patch_embed'):
            first_layer = net.patch_embed.proj
        else:
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
                if cov_mat is None:
                    cov_mat = A.clone()
                else:
                    cov_mat.add_(A)
            cov_mat /= n_batches

        # Compute the inverse preconditioner from the input covariance.
        self.first_layer = first_layer
        eye_matrix = torch.eye(cov_mat.size(0), device=device, dtype=cov_mat.dtype)
        self.input_cov_inv = torch.linalg.inv(cov_mat + self.damping * eye_matrix)
        self.model = net
        # Remove the forward hook on the first layer after configuration.
        self.layer_map[first_layer]['fwd_hook'].remove()

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

        actv = forward_input[0].data  # raw activation input

        # Handle token-wise statistics for ViT-style inputs.
        if actv.ndim == 3:  # Expected shape: [B, seq_length, hidden_dim]
            B, seq_length, hidden_dim = actv.shape
            if self.vit_mode:
                # Optionally use pre-LayerNorm activations: do not apply normalization here.
                # Separate the class token (usually at index 0) from patch tokens.
                cls_token = actv[:, 0, :]  # [B, hidden_dim]
                patch_tokens = actv[:, 1:, :]  # [B, seq_length-1, hidden_dim]
                cls_mean = cls_token.mean(0)
                patch_mean = patch_tokens.mean(0)
                # Weighted combination of the class token and patch tokens.
                avg_actv = self.cls_token_weight * cls_mean + (1.0 - self.cls_token_weight) * patch_mean
            else:
                # If not in ViT mode, flatten batch and sequence dimensions and average.
                actv = actv.view(-1, actv.size(-1))
                avg_actv = actv.mean(0)
        else:
            # For activations with other shapes, simply average over the batch dimension.
            avg_actv = actv.mean(0)

        # For Conv2d and Linear layers, if a bias exists, append a 1.
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            ones = torch.ones((avg_actv.shape[0],), device=avg_actv.device, dtype=avg_actv.dtype)
            avg_actv = torch.cat([avg_actv, ones], dim=0)

        # For LayerNorm modules, if not using pre-LN stats, perform normalization as in the original code.
        if isinstance(module, nn.LayerNorm) and not self.use_pre_ln_stats:
            if actv.ndim > 2:
                actv = actv.view(-1, actv.size(-1))
            mean = actv.mean(dim=-1, keepdim=True)
            var = actv.var(dim=-1, unbiased=False, keepdim=True)
            norm_actv = (actv - mean) / torch.sqrt(var + module.eps)
            avg_actv = norm_actv.mean(0)

        # Optionally, apply a nonlinear transformation (e.g. tanh) to adjust the activation statistics.
        if self.vit_mode and self.use_nonlinear_transform:
            avg_actv = torch.tanh(avg_actv)

        # Update the exponential moving average in the module’s state.
        state = self.state[module]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv, device=avg_actv.device)
        state['exp_avg'].mul_(stat_decay).add_(avg_actv, alpha=1 - stat_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        b_updated = False
        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = self.damping
        if self._step % self.Tinv == 0:
            b_updated = True
            self.emastep += 1

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.LayerNorm)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if layer == self.first_layer:
                    A_inv = self.input_cov_inv.to(grad_mat.dtype)
                else:
                    if b_updated:
                        bias_correction = 1.0 - (stat_decay ** self.emastep)
                        exp_avg = state['exp_avg'].div(bias_correction).to(grad_mat.dtype)
                        sq_norm = torch.linalg.norm(exp_avg).pow(2)

                        if 'A_inv' not in state:
                            state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype)
                        else:
                            state['A_inv'].copy_(torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype))

                        state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))
                    A_inv = state['A_inv'].to(grad_mat.dtype)

                if isinstance(layer, nn.LayerNorm):
                    # For LayerNorm, precondition only the weight.
                    v = A_inv @ grad_mat
                    layer.weight.grad.data.copy_(v.view_as(layer.weight))
                    # Leave layer.bias.grad unchanged.
                else:
                    # For Linear and Conv2d layers, precondition both weight and bias.
                    v = grad_mat @ A_inv
                    if layer.bias is not None:
                        v_weight = v[:, :-1]
                        v_bias = v[:, -1:]
                        layer.weight.grad.data.copy_(v_weight.view_as(layer.weight))
                        layer.bias.grad.data.copy_(v_bias.view_as(layer.bias))
                    else:
                        layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        momentum_step(self)
        self._step += 1

        return loss
