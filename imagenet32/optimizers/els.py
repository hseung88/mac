import logging as logger
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.opt_utils2 import extract_patches, reshape_grad, momentum_step, nag_step
from .utils.torch_utils import build_layer_map

class ELS(Optimizer):
    """
    Nystrom approximation based optimizer with activation-based leverage scoring sketching.
    """
    def __init__(self,
                 params,
                 lr=0.1,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.01,
                 weight_decay=0.0005,
                 Tcov=5,
                 Tinv=50,
                 rank_size=5,
                ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if Tcov > Tinv:
            raise ValueError(f"Tcov={Tcov:d} > Tinv={Tinv:d}")

        defaults = dict(lr=lr, damping=damping, momentum=momentum, weight_decay=weight_decay,
                        stat_decay=stat_decay, step=0, ema_step=0)
        super(ELS, self).__init__(params, defaults)

        self._model = None
        self.stat_decay = stat_decay
        self.Tcov = Tcov
        self.Tinv = Tinv
        self.rank_size = rank_size
        
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

    def _store_input(self, module, forward_input, forward_output):
        if not module.training or not torch.is_grad_enabled():
            return

        group = self.param_groups[0]
        step = group['step']
        if (step % self.Tcov) != 0:
            return

        stat_decay = group['stat_decay']
        actv = forward_input[0].detach().clone()

        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride,
                                   module.padding, depthwise)
        elif actv.ndim > 2:  # linear layers in transformers
            actv = actv.view(-1, actv.size(-1))

        if module.bias is not None:
            actv = torch.cat([actv, torch.ones((actv.size(0), 1), device=actv.device)], dim=1)

        p = actv.size(1)
        rank = min(self.rank_size, p)

        # Compute leverage scores
        leverage_scores = torch.sum(actv ** 2, dim=0)
        # Select top-k components based on leverage scores
        topk_indices = torch.topk(leverage_scores, rank, largest=True).indices

        C = actv[:, topk_indices]
        print(C.size())
        state = self.state[module]
        if 'ema_C' not in state:
            state['ema_C'] = torch.zeros_like(C, device=actv.device)

        state['ema_C'].mul_(stat_decay).add_(C, alpha=1.0 - stat_decay)

    @torch.no_grad()
    def update_inverse(self, layer, damping, bias_correction):
        state = self.state[layer]
        C = state['ema_C'].div(bias_correction)

        # Subsampled covariance
        C_sub = torch.matmul(C.t(), C / C.size(0))

        # Use efficient matrix multiplication and inversion
        identity_matrix = torch.eye(C_sub.size(0), device=C.device)
        C_sub_inv = torch.inverse(C_sub + damping * identity_matrix)
        C_sub_inv_CT = torch.matmul(C_sub_inv, C.t() / C.size(0))
        state['A_inv'] = torch.matmul(C, C_sub_inv_CT)

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

        b_inv_update = (group['step'] % self.Tinv) == 0
        group['step'] += 1

        bias_correction1 = 1.0 - (stat_decay ** group['ema_step'])

        for layer in self.layer_map:
            if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                continue

            if b_inv_update:
                self.update_inverse(layer, damping, bias_correction1)

            state = self.state[layer]
            grad_mat = reshape_grad(layer)
            v = torch.matmul(grad_mat, state['A_inv'])

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
