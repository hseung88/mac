import math
from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import reshape_grad, build_layer_map, update_step


class AdaNorm(Optimizer):
    def __init__(
            self,
            params,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=5e-4,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._model = None

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        beta1 = group['beta1']
        beta2 = group['beta2']
        eps = group['eps']

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if not state:
                    state['m'] = torch.zeros_like(grad_mat)
                    state['v'] = torch.zeros_like(grad_mat)
                    state['t'] = 0

                m, v = state['m'], state['v']
                state['t'] += 1
                t = state['t']

                m.mul_(beta1).add_(grad_mat, alpha=1 - beta1)

                #grad_mean_col, grad_mean_row = grad_mat.mean(0), grad_mat.mean(1)
                #grad_outer = torch.outer(grad_mean_row, grad_mean_col)
                #v.mul_(beta2).add_(grad_outer, alpha=1 - beta2)
                v.mul_(beta2).addcmul_(grad_mat, grad_mat, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                v = m_hat / (torch.sqrt(v_hat) + eps)

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        update_step(self)

        return loss
