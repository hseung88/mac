from typing import List
import torch
import torch.nn as nn
from torch.optim import Optimizer
import logging as log
from .utils.torch_utils import build_layer_map, trainable_modules
from .utils.opt_utils2 import extract_patches, reshape_grad, sgd_step, momentum_step, nag_step


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


class AdaActDR1(Optimizer):
    def __init__(
        self,
        params,
        lr=0.1,
        momentum=0.9,
        stat_decay=0.999,
        damping=1.0,
        eps=1.0,
        weight_decay=0.0005,
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
                        eps=eps,
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
                                         fwd_hook_fn=self._capture_activation,
                                         bwd_hook_fn=self._capture_backprop)

    def _configure(self, train_loader, net, device):
        n_batches = len(train_loader)
        cov_mat = None

        # this is not guranteed
        _, first_layer = next(trainable_modules(net))

        with torch.no_grad():
            for images, _ in train_loader:
                #batch_size = images.size(0)
                images = images.to(device)
                actv = extract_patches(images, first_layer.kernel_size,
                                       first_layer.stride, first_layer.padding,
                                       depthwise=False)
                if first_layer.bias is not None:
                    actv = torch.cat([actv, actv.new_ones((actv.size(0), 1))], dim=1)

                A = actv.t() @ (actv / actv.size(0))
                if cov_mat is None:
                    cov_mat = A
                else:
                    cov_mat.add_(A)

            cov_mat /= n_batches

        self.first_layer = first_layer
        self.input_cov_inv = torch.linalg.inv(
            cov_mat + self.damping * torch.eye(cov_mat.size(0), device=device)
        )
        self.model = net
        h_fwd_hook = self.layer_map[first_layer]['fwd_hook']
        h_fwd_hook.remove()

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
        corrected_stat_decay = stat_decay ** self.update_freq

        is_conv = isinstance(module, nn.Conv2d)
        actv = forward_input[0].detach().clone()
        #batch_size = actv.size(0)

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
        state['exp_avg'].mul_(corrected_stat_decay).add_(avg_actv, alpha=1.0 - corrected_stat_decay)

    def _capture_backprop(
        self,
        module: nn.Module,
        _grad_input: torch.Tensor,
        grad_output: torch.Tensor
    ):
        group = self.param_groups[0]
        step = group['step']
        if (step % self.update_freq) != 0:
            return

        stat_decay = group['stat_decay']
        corrected_stat_decay = stat_decay ** self.update_freq

        is_conv = isinstance(module, nn.Conv2d)
        g = grad_output[0].detach().clone()

        if is_conv:
            spatial_size = g.size(2) * g.size(3)
            g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if is_conv:
            g = g * spatial_size

        avg_dz = g.pow(2).mean(0)
        state = self.state[module]
        if 'exp_avg_z' not in state:
            state['exp_avg_z'] = torch.zeros_like(avg_dz)

        state['exp_avg_z'].mul_(corrected_stat_decay).add_(avg_dz, alpha=1.0 - corrected_stat_decay)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        corrected_stat_decay = stat_decay ** self.update_freq
        damping = group['damping']
        eps = group['eps']
        b_updated = False

        if (group['step'] % self.update_freq) == 0:
            group['ema_step'] += 1
            b_updated = True

        group['step'] += 1

        bias_correction1 = 1.0 - (corrected_stat_decay ** group['ema_step'])

        # compute the preconditioned gradient layer-by-layer
        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if b_updated:
                    exp_avg_z = state['exp_avg_z'].div(bias_correction1)
                    if 'Z_inv' not in state:
                        state['Z_inv'] = torch.diag(exp_avg_z.new_ones(exp_avg_z.size(0)))

                    Z = exp_avg_z.add_(eps).reciprocal_()
                    state['Z_inv'].diagonal().copy_(Z)
                Z_inv = state['Z_inv']

                if layer == self.first_layer:
                    A_inv = self.input_cov_inv
                else:
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

                v = Z_inv @ grad_mat @ A_inv

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
