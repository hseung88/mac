import logging as logger
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.adaact_utils import AdaActStats
from .utils.torch_utils import build_layer_map
from .utils.tensor_utils import reshape_grad, moving_average


class AdaAct(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        betas=(0.99, 0.9, 0.99),
        eps=1e-8,
        weight_decay=1e-5,
        update=1
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameter at index 2: {}'.format(betas[2]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if update < 1.0:
            raise ValueError("Invalid update period: {}".format(update))

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)
        super(AdaAct, self).__init__(params, defaults)

        self._model = None
        self.stats = AdaActStats()
        self.ema_A = {}
        self.A_inv = {}
        self.ema_n = 0
        self._step = 0
        self.update = update

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
        group = self.param_groups[0]
        betas = group['betas']

        eval_mode = (not module.training)
        if eval_mode or (not torch.is_grad_enabled()):
            return

        if (self._step % self.update) == 0:
            A = self.stats(module, forward_input[0].detach().clone())

            if self._step == 0:
                self.ema_A[module] = A.new_ones(A.size(0))
                self.A_inv[module] = torch.zeros_like(self.ema_A[module])

            nag_A = A.add((A - self.ema_A[module]).mul(betas[1]))
            moving_average(nag_A, self.ema_A[module], betas[2])

    def update_inverse(self, layer, eps):
        A = self.ema_A[layer]
        bias_corrected = A.div(self.ema_n).add_(eps)

        self.A_inv[layer].copy_(torch.rsqrt(bias_corrected))

    def _update_parameters(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            step_size = group['lr']
            betas = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0 and self._step >= 20 * self.update:
                    d_p.add_(p.data, alpha=weight_decay)

                if betas[0] != 0:
                    param_state = self.state[p]
                    if 'momentum_buff' not in param_state:
                        param_state['momentum_buff'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buff']
                        buf.mul_(betas[0]).add_(d_p)
                        d_p.add_(buf, alpha=betas[0])

                p.data.add_(d_p, alpha=-step_size)

    def step(self, closure=None):
        group = self.param_groups[0]
        eps = group['eps']
        betas = group['betas']

        if 'ema_step' not in group:
            group['ema_step'] = 0

        if (self._step % self.update) == 0:
            group['ema_step'] += 1
            self.ema_n = 1.0 - betas[2] ** group['ema_step']

        b_inv_update = ((self._step % self.update) == 0)

        # compute the preconditioned gradient layer-by-layer
        for layer in self.layer_map:

            if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                continue

            if b_inv_update:
                self.update_inverse(layer, eps)

            grad_mat = reshape_grad(layer)
            v = grad_mat.mul(self.A_inv[layer])

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
