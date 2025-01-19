from typing import List
import logging as log
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.torch_utils import build_layer_map, trainable_modules
from .utils.opt_utils2 import extract_patches, reshape_grad, sgd_step, momentum_step, nag_step
from . import FairseqOptimizer, register_optimizer


@register_optimizer("adaactr1")
class AdaActR1(FairseqOptimizer):
    def __init__(self, cfg, params):
        super().__init__(cfg)
        self._optimizer = AdaActR1(params, **self.optimizer_config)
    
    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum factor')
        parser.add_argument('--weight-decay', '--wd', default=0.01, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--stat_decay', default=0.95, type=float, help='stat decay')
        parser.add_argument('--damping', default=0.001, type=float, help='damping factor')
        parser.add_argument('--update', default=50, type=int, help='preconditioner update and inverse period')
        parser.add_argument('--sgd_momentum_type', default='heavyball', type=str)
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
            "update_freq": self.cfg.update,
            "sgd_momentum_type": self.cfg.sgd_momentum_type,
        }
    
    @property
    def supports_flat_params(self):
        return True
    
    
    
class AdaActR1(Optimizer):    
    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0.9,
        stat_decay=0.95,
        damping=0.001,
        weight_decay=0.01,
        update_freq=50,
        sgd_momentum_type='heavyball',
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if stat_decay < 0.0:
            raise ValueError(f'Invalid stat_decay value: {stat_decay}')
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, stat_decay=stat_decay,
                        damping=damping,
                        weight_decay=weight_decay,
                        step=0, ema_step=0)
        super().__init__(params, defaults)

        self._model = None
        self.damping = damping
        self.update_freq = update_freq
        self.sgd_momentum_type = sgd_momentum_type

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")

        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model,
                                         fwd_hook_fn=self._capture_activation)
    
    def _capture_activation(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor
    ):
        eval_mode = (not module.training)
        if eval_mode or (not torch.is_grad_enabled()):
            return

        group = self.param_groups[0]
        step = group['step']
        if (step % self.update_freq) != 0:
            return

        stat_decay = group['stat_decay']

        is_conv = isinstance(module, nn.Conv2d)
        actv = forward_input[0].detach().clone()
        # batch_size = actv.size(0)

        if is_conv:
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride,
                                   module.padding, depthwise)
        elif actv.ndim > 2:  # linear layers in transformers
            actv = actv.reshape(-1, actv.size(-1))
            
        if module.bias is not None:
            # bias trick
            actv = torch.cat([actv, actv.new_ones((actv.size(0), 1))], dim=1)

        avg_actv = actv.mean(0)
        
        state = self.state[module]
        if len(state) == 0:
            state['exp_avg'] = torch.zeros_like(avg_actv)
        
        # EMA
        state['exp_avg'].mul_(stat_decay).add_(avg_actv, alpha=1.0 - stat_decay)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = group['damping']
        b_updated = False

        if (group['step'] % self.update_freq) == 0:
            group['ema_step'] += 1
            b_updated = True

        group['step'] += 1

        bias_correction1 = 1.0 - (stat_decay ** group['ema_step'])

        # compute the preconditioned gradient layer-by-layer
        for layer in self.layer_map:
            #print(self.layer_map[layer])
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)
                
                # Check if 'exp_avg' is in state right before accessing it
                if 'exp_avg' not in state:
                    print(f"'exp_avg' not found for layer {layer}")
                    continue 
                
                if b_updated:
                    exp_avg = state['exp_avg'].div(bias_correction1)
                    sq_norm = torch.linalg.norm(exp_avg) ** 2
                    
                    if 'A_inv' not in state:
                        state['A_inv'] = torch.diag(exp_avg.new_ones(exp_avg.size(0)))
                    else:
                        state['A_inv'].copy_(torch.diag(exp_avg.new_ones(exp_avg.size(0))))

                    state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))
                    state['A_inv'].div_(damping)

                A_inv = state['A_inv']
                v = grad_mat @ A_inv

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
