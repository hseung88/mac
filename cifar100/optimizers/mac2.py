import math
from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, momentum_step, adam_step


class MAC2(Optimizer):
    def __init__(
            self,
            params,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            damping=1.0,
            weight_decay=0.01,
            Tcov=5,
            Tinv=50,
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
        self.damping = damping
        self.Tcov = Tcov
        self.Tinv = Tinv
        self._step = 0
        self.emastep = 0

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
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

        self.emastep += 1

        group = self.param_groups[0]
        beta2 = group['beta2']

        actv = forward_input[0].data
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:  # linear layers in transformers
                actv = actv.view(-1, actv.size(-1))

        if module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        mean_actv = actv.mean(0)
        #var_actv = torch.mean(actv.pow(2), axis=0)
        #var_actv = actv.var(dim=0, unbiased=True)
        var_actv = actv.var(dim=0)

        state = self.state[module]
        if 'exp_avg_actv' not in state:
            state['exp_avg_mean'] = torch.zeros_like(mean_actv, device=mean_actv.device)
            state['exp_avg_var'] = torch.zeros_like(var_actv, device=var_actv.device)
        state['exp_avg_mean'].mul_(beta2).add_(mean_actv, alpha=1 - beta2)
        state['exp_avg_var'].mul_(beta2).add_(var_actv, alpha=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        beta2 = group['beta2']
        damping = self.damping
        b_updated = (self._step % self.Tinv == 0)

        # Preconditioned SGD step
        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)
                #state['grad_mat'] = grad_mat

                if b_updated:
                    bias_correction = 1.0 - (beta2 ** self.emastep)
                    exp_avg_mean = state['exp_avg_mean'].div(bias_correction)
                    exp_avg_var = state['exp_avg_var'].div(bias_correction)
                    exp_avg_var += damping
                    m_div_v = exp_avg_mean / exp_avg_var
                    denominator = 1.0 + exp_avg_mean.dot(m_div_v)
                    correction = torch.outer(m_div_v, m_div_v) / denominator

                    state['A_inv'] = torch.diag(1.0 / exp_avg_var)
                    state['A_inv'].sub_(correction)

                    # sq_norm = torch.linalg.norm(exp_avg).pow(2)
                    #eye_matrix = torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype)

                    #state['A_inv'] = torch.outer(exp_avg, exp_avg)
                    #state['A_inv'].div_(sq_norm + damping)
                    #state['A_inv'].div_(damping)

                    #state['A_ortho_proj'] = eye_matrix.sub_(torch.outer(exp_avg, exp_avg))

                A_inv = state['A_inv']
                v = grad_mat @ A_inv

                #v = grad_mat.div(damping + torch.linalg.norm(grad_mat).pow(2)).div(damping) - grad_mat @ A_inv

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        momentum_step(self)
        """
        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = state['grad_mat']

                A_ortho_proj = state['A_ortho_proj']

                v = grad_mat @ A_ortho_proj

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        adam_step(self)
        """
        self._step += 1

        return loss
