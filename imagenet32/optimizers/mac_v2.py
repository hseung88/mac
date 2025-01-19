from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.torch_utils import build_layer_map
from .utils.opt_utils2 import extract_patches, reshape_grad, sgd_step

class MAC_V2(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        damping=1.0,
        weight_decay=0.01,
        Tcov=1,
        Tinv=5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas,
                        damping=damping, weight_decay=weight_decay,
                        step=0, ema_step=0)
        super().__init__(params, defaults)

        self._model = None
        self.Tcov = Tcov
        self.Tinv = Tinv

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

        group = self.param_groups[0]
        beta2 = group['betas'][1]
        
        if group['step'] % self.Tcov != 0:
            return

        actv = forward_input[0].data
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif actv.ndim > 2:
            actv = actv.reshape(-1, actv.size(-1))

        if module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(0)

        state = self.state[module]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv, device=actv.device, dtype=actv.dtype)

        state['exp_avg'].mul_(beta2).add_(avg_actv, alpha=1.0 - beta2)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        beta1, beta2 = group['betas']
        damping = group['damping']
        
        b_updated = False

        if group['step'] % self.Tinv == 0:
            group['ema_step'] += 1
            b_updated = True

        group['step'] += 1

        bias_correction1 = 1.0 - (beta1 ** group['step'])
        bias_correction2 = 1.0 - (beta2 ** group['ema_step'])

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)
                
                # Initialize EMA for gradient if not present
                if 'exp_avg_grad' not in state:
                    state['exp_avg_grad'] = torch.zeros_like(grad_mat)

                state['exp_avg_grad'].mul_(beta1).add_(grad_mat, alpha=1.0 - beta1)
                exp_avg_grad = state['exp_avg_grad'].div(bias_correction1)
                
                if b_updated:
                    exp_avg = state['exp_avg'].div(bias_correction2)
                    sq_norm = torch.linalg.norm(exp_avg).pow(2)

                    if 'A_inv' not in state:
                        state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype)
                    else:
                        state['A_inv'].copy_(torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype))

                    state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))
                    state['A_inv'].div_(damping)

                A_inv = state['A_inv']

                v = exp_avg_grad @ A_inv

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        sgd_step(self)

        return loss
