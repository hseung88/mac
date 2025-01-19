import torch
import torch.nn as nn
from torch.optim import Optimizer
import logging as logger
from .utils.foof_utils import FoofStats
from .utils.torch_utils import build_layer_map
from .utils.tensor_utils import reshape_grad, moving_average

class FOOF(Optimizer):
    def __init__(self,
                 params,
                 lr=0.1,
                 momentum=0.95,
                 stat_decay=0.99,
                 damping=0.01,
                 weight_decay=1e-4,
                 Tcov=5,
                 Tinv=50):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if Tcov > Tinv:
            raise ValueError(f"Tcov={Tcov:d} > Tinv={Tinv:d}")

        defaults = dict(lr=lr, damping=damping, momentum=momentum,
                        weight_decay=weight_decay)
        super(FOOF, self).__init__(params, defaults)

        self._model = None
        self.stats = FoofStats()
        self.ema_A = {}
        self.A_inv = {}
        self._step = 0
        self._emastep = 0

        self.stat_decay = stat_decay
        self.Tcov = Tcov
        self.Tinv = Tinv

    @property
    def model(self):
        if self._model is None:
            logger.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self._prepare_model()

    def _prepare_model(self):
        self.layer_map = build_layer_map(self._model, fwd_hook_fn=self._store_input)

    def _store_input(self, module, forward_input, forward_output):
        if not module.training or not torch.is_grad_enabled():
            return

        if (self._step % self.Tcov) == 0:
            actv = forward_input[0].detach()
            A = self.stats(module, actv)

            if self._step == 0:
                self.ema_A[module] = torch.zeros_like(A)

            moving_average(A, self.ema_A[module], self.stat_decay)

    def update_inverse(self, layer, damping):
        A = self.ema_A[layer]
        correction = 1.0 - (self.stat_decay ** self._emastep)

        damped_A = A / correction + damping * torch.eye(A.size(0), device=A.device)
        self.A_inv[layer] = torch.linalg.inv(damped_A)

    def _update_parameters(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            step_size = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    buf = param_state.get('momentum_buffer', torch.zeros_like(p))

                    buf.mul_(momentum).add_(d_p)
                    d_p = buf
                    param_state['momentum_buffer'] = buf

                p.data.add_(d_p, alpha=-step_size)

    def step(self, closure=None):
        if closure is not None:
            closure()

        group = self.param_groups[0]
        damping = group['damping']

        if (self._step % self.Tcov) == 0:
            self._emastep += 1

        b_inv_update = (self._step % self.Tinv) == 0

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                if b_inv_update:
                    self.update_inverse(layer, damping)

                grad_mat = reshape_grad(layer)
                v = torch.matmul(grad_mat, self.A_inv[layer])

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        self._update_parameters()
        self._step += 1
