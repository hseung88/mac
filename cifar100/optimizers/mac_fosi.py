import math
from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, trainable_modules, update_step

class MACFOSI(Optimizer):
    def __init__(
            self,
            params,
            lr=0.1,
            momentum=0.9,
            stat_decay=0.95,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            damping=1.0,
            weight_decay=5e-4,
            Tcov=50,
            Tinv=50,
            lr_scale=100,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr,
                        momentum=momentum,
                        stat_decay=stat_decay,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._model = None
        self.damping = damping
        self.Tcov = Tcov
        self.Tinv = Tinv
        self._step = 0
        self.emastep = 0
        self.lr_scale = lr_scale

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

        if module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(0)

        state = self.state[module]
        if 'exp_avg_actv' not in state:
            state['exp_avg_actv'] = torch.zeros_like(avg_actv, device=avg_actv.device)
        state['exp_avg_actv'].mul_(stat_decay).add_(avg_actv, alpha=1 - stat_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        momentum = group['momentum']
        stat_decay = group['stat_decay']
        beta1 = group['beta1']
        beta2 = group['beta2']
        eps = group['eps']
        damping = self.damping

        b_updated = (self._step % self.Tinv == 0)

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                # Preconditioning step
                bias_correction = 1.0 - (stat_decay ** self.emastep)
                exp_avg_actv = state['exp_avg_actv'].div(bias_correction)
                if b_updated:
                    sq_norm = torch.linalg.norm(exp_avg_actv).pow(2)

                    state['A_inv'] = torch.outer(exp_avg_actv, exp_avg_actv)
                    #state['A_inv'].mul_(sq_norm).div_(sq_norm + damping)
                    state['A_inv'].div_(sq_norm + damping)

                A_inv = state['A_inv']

                d1 = grad_mat @ A_inv * self.lr_scale

                #if 'momentum_buffer' not in state:
                #    state['momentum_buffer'] = torch.zeros_like(d1, device=d1.device)
                #d_heavyball = state['momentum_buffer'].mul_(momentum).add_(d1)
                d_heavyball = d1

                # Base optimizer step (Adam)
                eye_matrix = torch.eye(grad_mat.size(1), device=grad_mat.device, dtype=grad_mat.dtype)
                project_mat = eye_matrix.sub_(torch.outer(exp_avg_actv, exp_avg_actv))
                grad_mat_proj = grad_mat @ project_mat

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad_mat_proj, device=grad_mat_proj.device)
                    state['exp_avg_sq'] = torch.zeros_like(grad_mat_proj, device=grad_mat_proj.device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad_mat_proj, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_mat_proj, grad_mat_proj, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1.0 - beta1 ** (self._step + 1)
                bias_correction2 = 1.0 - beta2 ** (self._step + 1)

                grad_mat_proj_adam = (exp_avg / denom) * (math.sqrt(bias_correction2) / bias_correction1)
                d_adam = grad_mat_proj_adam @ project_mat

                v = d_heavyball + d_adam

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        update_step(self)
        self._step += 1

        return loss
