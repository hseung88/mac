import math
from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, trainable_modules, momentum_step, nag_step


class MAC(Optimizer):
    def __init__(
            self,
            params,
            lr=0.1,
            momentum=0.9,
            stat_decay=0.95,
            damping=1e-8,
            weight_decay=5e-4,
            Tcov=5,
            Tinv=50,
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

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model, fwd_hook_fn=self._capture_activation)

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

        self.emastep += 1

        group = self.param_groups[0]
        stat_decay = group['stat_decay']

        actv = forward_input[0].data
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:  # linear layers in transformers
                actv = actv.view(-1, actv.size(-1))
        elif isinstance(module, nn.LayerNorm):
            if actv.ndim > 2:
                actv = actv.view(-1, actv.size(-1))
            # Standardize the inputs to mimic the behavior of LayerNorm's internal normalization.
            # Compute the mean and variance along the last dimension (features)
            mean = actv.mean(dim=-1, keepdim=True)
            var = actv.var(dim=-1, unbiased=False, keepdim=True)
            actv = (actv - mean) / torch.sqrt(var + self.damping)

        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(0)

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

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = self.damping
        b_updated = (self._step % self.Tinv == 0)

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.LayerNorm)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if b_updated:
                    bias_correction = 1.0 - (stat_decay ** self.emastep)
                    exp_avg = state['exp_avg'].div(bias_correction)
                    sq_norm = torch.linalg.norm(exp_avg).pow(2)

                    if 'A_inv' not in state:
                        state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype)
                    else:
                        state['A_inv'].copy_(torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype))

                    state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))
                    #state['A_inv'].div_(damping)

                A_inv = state['A_inv'].to(grad_mat.dtype)

                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    v = grad_mat @ A_inv
                #else:
                #    v = A_inv @ grad_mat

                if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.bias is not None:
                    # For Linear/Conv2d, we previously concatenated bias into grad_mat.
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                elif isinstance(layer, nn.LayerNorm) and layer.bias is not None:
                    # For LayerNorm, weight and bias are separate 1D parameters.
                    # Compute preconditioning separately on each.
                    # Make sure the preconditioning matrix A_inv has shape (n, n) where n == layer.weight.shape[0].
                    # Compute update for weight:
                    weight_grad = layer.weight.grad.data
                    precond_weight = A_inv @ weight_grad
                    layer.weight.grad.data.copy_(precond_weight.view_as(weight_grad))
                    # Compute update for bias:
                    bias_grad = layer.bias.grad.data
                    precond_bias = A_inv @ bias_grad
                    layer.bias.grad.data.copy_(precond_bias.view_as(bias_grad))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        nag_step(self)
        self._step += 1

        return loss
