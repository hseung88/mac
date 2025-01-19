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
                 lr=0.01,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.01,
                 weight_decay=1e-5,
                 Tcov=5,
                 Tinv=50):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if Tcov > Tinv:
            raise ValueError("Tcov={Tcov:d} < Tinv={Tinv:d}")

        defaults = dict(lr=lr, damping=damping, momentum=momentum,
                        weight_decay=weight_decay)
        super(FOOF, self).__init__(params, defaults)

        self._model = None
        self.stats = FoofStats()
        self.ema_A = {}
        self.A_inv = {}
        self.ema_n = 0
        self._step = 0

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
        self.layer_map = build_layer_map(self._model,
                                         fwd_hook_fn=self._store_input)

    def _store_input(self, module, forward_input, forward_output):
        eval_mode = (not module.training)
        if eval_mode or (not torch.is_grad_enabled()):
            return

        if (self._step % self.Tcov) == 0:
            A = self.stats(module, forward_input[0].detach().clone())

            if self._step == 0:
                self.ema_A[module] = torch.diag(A.new_ones(A.size(0)))
                # self.ema_A[module] = torch.zeros_like(A)

            moving_average(A, self.ema_A[module], self.stat_decay)

    def update_inverse(self, layer, damping):
        # compute damping factor
        A = self.ema_A[layer]
        correction = 1.0 / self.ema_n

        damped_A = correction * A + damping * torch.diag(A.new_ones(A.size(0)))
        self.A_inv[layer] = torch.inverse(damped_A)

    def _update_parameters(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            step_size = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0 and self._step >= 20 * self.Tcov:
                    d_p.add_(p.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p)

                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-step_size)

    def step(self, closure=None):
        group = self.param_groups[0]
        damping = group['damping']

        if (self._step % self.Tcov) == 0:
            self.ema_n *= self.stat_decay
            self.ema_n += (1.0 - self.stat_decay)

        b_inv_update = ((self._step % self.Tinv) == 0)

        # compute the preconditioned gradient layer-by-layer
        for layer in self.layer_map:

            if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                continue

            if b_inv_update:
                self.update_inverse(layer, damping)

            grad_mat = reshape_grad(layer)
            v = grad_mat @ self.A_inv[layer]

            if layer.bias is not None:
                v = [v[:, :-1], v[:, -1:]]
                v[0] = v[0].view_as(layer.weight)
                v[1] = v[1].view_as(layer.bias)

                layer.weight.grad.data.copy_(v[0])
                layer.bias.grad.data.copy_(v[1])
            else:
                v = v.view(layer.weight.grad.size())
                layer.weight.grad.data.copy_(v)

        self._update_parameters()
        self._step += 1
