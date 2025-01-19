from typing import List
import logging as logger
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.torch_utils import build_layer_map
from .utils.tensor_utils import moving_average
from .utils.opt_utils2 import extract_patches, reshape_grad, momentum_step

class Shaper(Optimizer):
    def __init__(self,
                 params,
                 lr=0.1,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.01,
                 weight_decay=0.0005,
                 Tcov=5,
                 Tinv=50):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if Tcov > Tinv:
            raise ValueError(f"Tcov={Tcov} must be less than or equal to Tinv={Tinv}")

        defaults = dict(lr=lr, momentum=momentum, stat_decay=stat_decay,
                        damping=damping, weight_decay=weight_decay, 
                        step=0, ema_step=0)
        super(Shaper, self).__init__(params, defaults)

        self._model = None
        self.Tcov = Tcov
        self.Tinv = Tinv
        self.ema_a = {}
        self.ema_A = {}
        self.A_inv = {}

    @property
    def model(self):
        if self._model is None:
            logger.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(self._model,
                                         fwd_hook_fn=self._store_input,
                                         supported_layers=(nn.Linear, nn.Conv2d))

    def _store_input(self,
                     module: nn.Module,
                     forward_input: List[torch.Tensor],
                     _forward_output: torch.Tensor):
        if not module.training or not torch.is_grad_enabled():
            return

        group = self.param_groups[0]
        step = group['step']

        if (step % self.Tcov) != 0:
            return

        stat_decay = group['stat_decay']
        
        actv = forward_input[0].detach().clone()
        is_conv = isinstance(module, nn.Conv2d)

        if is_conv:
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride,
                                   module.padding, depthwise)
        elif actv.ndim > 2:
            actv = actv.reshape(-1, actv.size(-1))

        if module.bias is not None:
            actv = torch.cat([actv, actv.new_ones((actv.size(0), 1))], dim=1)

        A = torch.matmul(actv.t(), actv / actv.size(0))

        if module not in self.ema_A:
            self.ema_A[module] = torch.zeros_like(A)

        moving_average(A, self.ema_A[module], stat_decay)

        if module not in self.ema_a:
            self.ema_a[module] = torch.zeros_like(actv[0])

        moving_average(actv.mean(0), self.ema_a[module], stat_decay)

    def bfgs_update(self, layer, damping):
        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        bias_correction = 1.0 - (stat_decay ** group['ema_step'])

        A = self.ema_A[layer].div(bias_correction)
        a = self.ema_a[layer].div(bias_correction)
        damped_A = A + damping * torch.eye(A.size(0), device=A.device)

        if layer not in self.A_inv:
            self.A_inv[layer] = torch.eye(A.size(0), device=A.device)

        s = torch.matmul(self.A_inv[layer], a)
        y = torch.matmul(damped_A, s)
        Hy = torch.matmul(self.A_inv[layer], y)

        yHy = torch.dot(y, Hy)
        rho = 1. / torch.dot(s, y)

        if rho.abs() < 1e-8:
            logger.info("The value of rho is too close to zero.")
            return

        H_new = self.A_inv[layer] + ((rho**2) * yHy + rho) * torch.outer(s, s)
        H_new -= rho * (torch.outer(s, Hy) + torch.outer(Hy, s))

        if torch.max(torch.isinf(H_new)):
            logger.critical("The updated Hessian has an inf value.")
            raise RuntimeError("The updated Hessian contains INF values.")

        self.A_inv[layer] = H_new

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        damping = group['damping']

        b_inv_update = (group['step'] % self.Tinv) == 0
        if (group['step'] % self.Tcov) == 0:
            group['ema_step'] += 1

        group['step'] += 1

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                if b_inv_update:
                    self.bfgs_update(layer, damping)

                grad = reshape_grad(layer)
                v = torch.matmul(grad, self.A_inv[layer])

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    v[0] = v[0].view_as(layer.weight)
                    v[1] = v[1].view_as(layer.bias)

                    layer.weight.grad.data.copy_(v[0])
                    layer.bias.grad.data.copy_(v[1])
                else:
                    v = v.view(layer.weight.grad.size())
                    layer.weight.grad.data.copy_(v)

        momentum_step(self)

        return loss
