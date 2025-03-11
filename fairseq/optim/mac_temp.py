import math
import logging
from typing import Any, List
import torch
import torch.nn as nn
from torch.optim import Optimizer
from fairseq.optim import FairseqOptimizer, register_optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, momentum_step


logger = logging.getLogger(__name__)


@register_optimizer('mac')
class MAC(FairseqOptimizer):
    def __init__(self, cfg, params):
        super().__init__(cfg)
        self._optimizer = MAC(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum factor')
        parser.add_argument('--stat_decay', default=0.95, type=float, metavar='SD',
                            help='stat decay')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--damping', default=1.0, type=float, metavar='DAMPING',
                            help='damping')
        parser.add_argument('--tcov', default=5, type=int, metavar='TCOV',
                            help='covariance update freq')
        parser.add_argument('--tinv', default=50, type=int, metavar='TINV',
                            help='inverse update freq')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0],
            "momentum": self.cfg.momentum,
            "stat_decay": self.cfg.stat_decay,
            "weight_decay": self.cfg.weight_decay,
            "damping": self.cfg.damping,
            "Tcov": self.cfg.tcov,
            "Tinv": self.cfg.tinv,
        }


class MAC(Optimizer):
    def __init__(
            self,
            params,
            lr=0.1,
            momentum=0.9,
            stat_decay=0.95,
            damping=1.0,
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
            logging.error("Model is not attached to the optimizer.")
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

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        name = self.layer_map[module]['name']

        actv = forward_input[0].detach().clone()
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:  # Expect shape [B, N, d] for transformer inputs.
                if "self_attn.v_proj" in name:
                    B, N, D = actv.shape
                actv = actv.reshape(-1, actv.size(-1))

        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(dim=0)

        #state = self.state[module]
        key = id(module)
        state = self.state.setdefault(key, {})
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv, device=avg_actv.device)
        state['exp_avg'].mul_(stat_decay).add_(avg_actv, alpha=1 - stat_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        b_updated = False
        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = self.damping
        if self._step % self.Tinv == 0:
            b_updated = True
            self.emastep += 1

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                #state = self.state[layer]
                key = id(layer)  # assuming 'layer' is the module
                state = self.state.get(key, {})
                grad_mat = reshape_grad(layer)
                print(layer)
                print(state)
                print(self._step % self.Tinv)
                print(b_updated)

                if b_updated:
                    bias_correction = 1.0 - (stat_decay ** self.emastep)
                    exp_avg = state['exp_avg'].div(bias_correction).to(grad_mat.dtype)
                    sq_norm = torch.dot(exp_avg, exp_avg)

                    if 'A_inv' not in state:
                        state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype)
                    else:
                        state['A_inv'].copy_(
                            torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype))

                    state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))
                    # state['A_inv'].div_(damping)

                A_inv = state['A_inv'].to(grad_mat.dtype)

                v = grad_mat @ A_inv

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        momentum_step(self)
        self._step += 1

        return loss