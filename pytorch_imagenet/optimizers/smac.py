from typing import List
import torch
import torch.nn as nn
from torch.optim import Optimizer
import logging as log
from .utils.torch_utils import build_layer_map, trainable_modules
from .utils.opt_utils2 import extract_patches, reshape_grad, sgd_step, momentum_step, nag_step

def try_contiguous(x):
    return x if x.is_contiguous() else x.contiguous()

class SMAC(Optimizer):
    def __init__(
        self,
        params,
        lr=0.1,
        momentum=0.9,
        stat_decay=0.999,
        damping=1.0,
        weight_decay=0.0005,
        update_freq=50,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if stat_decay < 0.0:
            raise ValueError(f"Invalid stat_decay value: {stat_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, stat_decay=stat_decay,
                        damping=damping, weight_decay=weight_decay, step=0, ema_step=0)
        super().__init__(params, defaults)

        self._model = None
        self.update_freq = update_freq

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model,
                                         fwd_hook_fn=self._capture_activation,
                                         bwd_hook_fn=self._capture_backprop)

    def _configure(self, train_loader, net, device):
        n_batches = len(train_loader)
        cov_mat = None

        _, first_layer = next(trainable_modules(net))

        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(device)
                actv = extract_patches(images, first_layer.kernel_size,
                                       first_layer.stride, first_layer.padding,
                                       depthwise=False)
                if first_layer.bias is not None:
                    actv = torch.cat([actv, torch.ones((actv.size(0), 1), device=actv.device)], dim=1)

                A = torch.matmul(actv.t(), actv) / actv.size(0)
                if cov_mat is None:
                    cov_mat = A
                else:
                    cov_mat.add_(A)

            cov_mat /= n_batches

        self.first_layer = first_layer
        self.input_cov_inv = torch.linalg.inv(
            cov_mat + self.defaults['damping'] * torch.eye(cov_mat.size(0), device=device)
        )
        self.model = net
        self.layer_map[first_layer]['fwd_hook'].remove()

    def _capture_activation(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor
    ):
        if not module.training or not torch.is_grad_enabled():
            return

        group = self.param_groups[0]
        if group['step'] % self.update_freq != 0:
            return

        actv = forward_input[0].detach().clone()
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif actv.ndim > 2:  # linear layers in transformers
            actv = actv.reshape(-1, actv.size(-1))

        if module.bias is not None:
            actv = torch.cat([actv, torch.ones((actv.size(0), 1), device=actv.device)], dim=1)

        avg_actv = actv.mean(0)
        state = self.state[module]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv)

        state['exp_avg'].mul_(group['stat_decay']).add_(avg_actv, alpha=1.0 - group['stat_decay'])

    def _capture_backprop(
        self,
        module: nn.Module,
        _grad_input: torch.Tensor,
        grad_output: torch.Tensor
    ):
        group = self.param_groups[0]
        if group['step'] % self.update_freq != 0:
            return

        g = grad_output[0].detach().clone()
        if isinstance(module, nn.Conv2d):
            spatial_size = g.size(2) * g.size(3)
            g = g.transpose(1, 2).transpose(2, 3).reshape(-1, g.size(1))
            g *= spatial_size
        else:
            g = g.view(-1, g.size(-1))

        avg_dz = g.pow(2).mean(0)
        state = self.state[module]
        if 'exp_avg_z' not in state:
            state['exp_avg_z'] = torch.zeros_like(avg_dz)

        state['exp_avg_z'].mul_(group['stat_decay']).add_(avg_dz, alpha=1.0 - group['stat_decay'])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = group['damping']
        b_updated = False

        if group['step'] % self.update_freq == 0:
            group['ema_step'] += 1
            b_updated = True

        group['step'] += 1
        bias_correction1 = 1.0 - (stat_decay ** group['ema_step'])

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if b_updated:
                    exp_avg_z = state['exp_avg_z'].div(bias_correction1)
                    Z = exp_avg_z.add_(damping).reciprocal_()
                    if 'Z_inv' not in state:
                        state['Z_inv'] = torch.diag(Z)
                    else:
                        state['Z_inv'].diagonal().copy_(Z)

                Z_inv = state['Z_inv']

                if layer == self.first_layer:
                    A_inv = self.input_cov_inv
                else:
                    if b_updated:
                        exp_avg = state['exp_avg'].div(bias_correction1)
                        sq_norm = exp_avg.norm().square()

                        if 'A_inv' not in state:
                            state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device)

                        A_inv = torch.eye(exp_avg.size(0), device=exp_avg.device) - torch.outer(exp_avg, exp_avg) / (damping + sq_norm)
                        A_inv /= damping
                        state['A_inv'].copy_(A_inv)

                    A_inv = state['A_inv']

                v = Z_inv @ grad_mat @ A_inv

                if layer.bias is not None:
                    v_weight, v_bias = v[:, :-1], v[:, -1:]
                    layer.weight.grad.copy_(v_weight.view_as(layer.weight))
                    layer.bias.grad.copy_(v_bias.view_as(layer.bias))
                else:
                    layer.weight.grad.copy_(v.view_as(layer.weight.grad))

        momentum_step(self)

        return loss
