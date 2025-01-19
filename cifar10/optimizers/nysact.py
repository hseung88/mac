#import logging as logger
#from tasks.base import MLTask
#from training.trainer import Trainer
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.opt_utils2 import extract_patches, reshape_grad, sgd_step, momentum_step, nag_step
from .utils.torch_utils import build_layer_map


class NysAct(Optimizer):
    def __init__(self,
                 params,
                 lr=0.1,
                 momentum=0.9,
                 stat_decay=0.99,
                 damping=1.0,
                 weight_decay=0.0005,
                 Tcov=5,
                 Tinv=50,
                 rank=5,
                 sketch_method='subcolumns',
                 sgd_momentum_type='heavyball'):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if Tcov > Tinv:
            raise ValueError("Tcov={Tcov:d} > Tinv={Tinv:d}")

        defaults = dict(lr=lr, damping=damping, momentum=momentum, weight_decay=weight_decay,
                        stat_decay=stat_decay, step=0, ema_step=0)
        super(NysAct, self).__init__(params, defaults)

        self._model = None
        self.stat_decay = stat_decay
        self.Tcov = Tcov
        self.Tinv = Tinv
        self.rank = rank
        self.sketch_method = sketch_method
        self.sgd_momentum_type = sgd_momentum_type

        self.sketch_mat = {}

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
                                         fwd_hook_fn=self._store_input,
                                         supported_layers=(nn.Linear, nn.Conv2d))

    #def _configure(self, trainer: Trainer, model: MLTask):
    #    self.model = model

    def _store_input(self, module, forward_input, forward_output):
        eval_mode = (not module.training)
        if eval_mode or (not torch.is_grad_enabled()):
            return

        group = self.param_groups[0]
        step = group['step']
        if (step % self.Tcov) != 0:
            return

        stat_decay = group['stat_decay']
        actv = forward_input[0].detach().clone()

        if isinstance(module, nn.Conv2d):
            # a.shape = B x out_H x out_W x (in_C * kernel_H * kernel_W)
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride,
                                   module.padding, depthwise)
            actv = actv.view(-1, actv.size(-1))
        elif actv.ndim > 2:  # linear layers in transformers
            actv = actv.view(-1, actv.size(-1))

        batch_size = actv.size(0)
        if module.bias is not None:
            actv = torch.cat([actv, actv.new_ones((batch_size, 1))], 1)

        p = actv.size(1)  # size of activations
        rank = min(self.rank, p)
        if rank == p or self.sketch_method == 'full':
            C = actv.t() @ (actv / batch_size)
        else:
            match self.sketch_method:
                case 'subcolumns':
                    # if module in self.col_indices:
                    #     indices = self.col_indices[module]
                    # else:
                    indices = torch.randperm(p)[:rank]
                    #self.sketch_mat[module] = indices.to(self.model.device)
                    self.sketch_mat[module] = indices.to(next(self.model.parameters()).device)
                    C = actv.t() @ (actv[:, indices] / batch_size)
                case 'gaussian':
                    #S = torch.randn(p, rank).to(self.model.device) / (p ** 0.5)
                    S = torch.randn(p, rank).to(next(self.model.parameters()).device) / (p ** 0.5)
                    S, _ = torch.linalg.qr(S, mode='reduced')
                    self.sketch_mat[module] = S
                    C = actv.t() @ (actv @ S) / batch_size
                case _:
                    raise NotImplementedError(f'unknown sketch method: {self.sketch_method}')

        state = self.state[module]
        if len(state) == 0:
            state['ema_C'] = torch.zeros_like(C)

        state['ema_C'].mul_(stat_decay).add_(C, alpha=1.0 - stat_decay)

    def update_inverse(self, layer, damping, bias_correction):
        # compute damping factor
        state = self.state[layer]
        C = state['ema_C'].div(bias_correction)

        if layer in self.sketch_mat:
            match self.sketch_method:
                case 'subcolumns':
                    indices = self.sketch_mat[layer]
                    W = C.take_along_dim(indices[None], dim=0)
                case 'gaussian':
                    W = self.sketch_mat[layer].t() @ C
                case _:
                    raise NotImplementedError(f'unknown sketch method: {self.sketch_method}')

            A_inv = C @ torch.inverse(C.t() @ C + damping * W) @ C.t()
            A_inv = torch.diag(C.new_ones(A_inv.size(0))) - A_inv
            state['A_inv'] = A_inv.div(damping)
        else:
            damped_C = C + damping * torch.diag(C.new_ones(C.size(0)))
            state['A_inv'] = torch.inverse(damped_C)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = group['damping']

        if (group['step'] % self.Tcov) == 0:
            group['ema_step'] += 1

        b_inv_update = ((group['step'] % self.Tinv) == 0)
        group['step'] += 1

        bias_correction1 = 1.0 - (stat_decay ** group['ema_step'])

        # compute the preconditioned gradient layer-by-layer
        for layer in self.layer_map:
            if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                continue

            if b_inv_update:
                self.update_inverse(layer, damping, bias_correction1)

            state = self.state[layer]
            grad_mat = reshape_grad(layer)
            v = grad_mat @ state['A_inv']
            # v = (grad_mat - grad_mat @ state['A_inv']).div(damping)

            if layer.bias is not None:
                v = [v[:, :-1], v[:, -1:]]
                v[0] = v[0].view_as(layer.weight)
                v[1] = v[1].view_as(layer.bias)

                layer.weight.grad.data.copy_(v[0])
                layer.bias.grad.data.copy_(v[1])
            else:
                v = v.view(layer.weight.grad.size())
                layer.weight.grad.data.copy_(v)

        if self.sgd_momentum_type == "heavyball":
            momentum_step(self)
        elif self.sgd_momentum_type == "nag":
            nag_step(self)
        else:
            sgd_step(self)

        return loss
