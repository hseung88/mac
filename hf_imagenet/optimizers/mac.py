from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, momentum_step, nag_step

class MAC(Optimizer):
    def __init__(
        self,
        params,
        lr=0.1,
        momentum=0.9,
        stat_decay=0.95,
        damping=5.0,
        weight_decay=0.0001,
        Tcov=5,
        Tinv=50,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if stat_decay < 0.0:
            raise ValueError(f'Invalid stat_decay value: {stat_decay}')
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, 
                        momentum=momentum,
                        stat_decay=stat_decay,
                        damping=damping,
                        weight_decay=weight_decay,)
        super().__init__(params, defaults)

        self._model = None
        self.stat_decay = stat_decay
        self.damping = damping
        self._step = 0
        self.ema_step = 0
        self.Tcov = Tcov
        self.Tinv = Tinv

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model, fwd_hook_fn=self._capture_activation)

    def _configure(self, train_loader, net, device):
        n_batches = len(train_loader)
        cov_mat = None

        #_, first_layer = next(trainable_modules(net))
        # Handle the case when the model is wrapped in DistributedDataParallel
        if hasattr(net, 'module'):
            net = net.module

        # Directly capture the first layer (patch embedding) of ViTs
        first_layer = net.patch_embed.proj

        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(device, non_blocking=True)
                actv = extract_patches(images, first_layer.kernel_size,
                                       first_layer.stride, first_layer.padding,
                                       depthwise=False)
                if first_layer.bias is not None:
                    ones = actv.new_ones((actv.shape[0], 1))
                    actv = torch.cat([actv, ones], dim=1)

                #A = torch.einsum('ij,jk->ik', actv.t(), actv) / actv.size(0)  # Optimized matrix multiplication
                A = torch.matmul(actv.t(), actv) / actv.size(0)
                if cov_mat is None:
                    cov_mat = A
                else:
                    cov_mat.add_(A)

            cov_mat /= n_batches

        self.first_layer = first_layer
        eye_matrix = torch.eye(cov_mat.size(0), device=device, dtype=cov_mat.dtype)
        #self.input_cov_inv = torch.linalg.inv(cov_mat + self.damping * eye_matrix)
        self.input_cov_inv = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat + self.damping * eye_matrix))
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

        if self._step % self.Tcov != 0:
            return

        actv = forward_input[0].data
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:  # linear layers in transformers
                actv = actv.view(-1, actv.size(-1))
        elif isinstance(module, nn.LayerNorm):
            actv = actv.view(-1, actv.size(-1))

        if module.bias is not None:
            if isinstance(module, nn.LayerNorm):
                pass
            else:
                ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
                actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(0)

        state = self.state[module]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv)

        state['exp_avg'].mul_(self.stat_decay).add_(avg_actv, alpha=1.0 - self.stat_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group['lr']
        weight_decay = group['weight_decay']
        damping = self.damping
        b_updated = False

        if self._step % self.Tinv == 0:
            self.ema_step += 1
            b_updated = True

        bias_correction = 1.0 - (self.stat_decay ** self.ema_step)

        # AdamW hyperparams for LN
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        for layer in self.layer_map:
            if not hasattr(layer, 'weight') or layer.weight.grad is None:
                continue

            if isinstance(layer, nn.LayerNorm):
                # typical AdamW states: exp_avg, exp_avg_sq for weight & bias
                ln_state = self.state[layer]

                # --- Weight AdamW ---
                w_grad = layer.weight.grad
                if 'exp_avg_w' not in ln_state:
                    ln_state['exp_avg_w'] = torch.zeros_like(layer.weight)
                    ln_state['exp_avg_sq_w'] = torch.zeros_like(layer.weight)

                exp_avg_w = ln_state['exp_avg_w']
                exp_avg_sq_w = ln_state['exp_avg_sq_w']

                # Exponential moving averages of gradient values
                exp_avg_w.mul_(beta1).add_(w_grad, alpha=1 - beta1)
                exp_avg_sq_w.mul_(beta2).addcmul_(w_grad, w_grad, value=1 - beta2)

                denom_w = exp_avg_sq_w.sqrt().add_(eps)
                step_size = lr
                # Param update
                layer.weight.data.addcdiv_(exp_avg_w, denom_w, value=-step_size)

                # Weight decay for LN weight
                if weight_decay != 0.0:
                    layer.weight.data.mul_(1 - lr * weight_decay)

                # --- Bias AdamW (if present) ---
                if layer.bias is not None and layer.bias.grad is not None:
                    b_grad = layer.bias.grad
                    if 'exp_avg_b' not in ln_state:
                        ln_state['exp_avg_b'] = torch.zeros_like(layer.bias)
                        ln_state['exp_avg_sq_b'] = torch.zeros_like(layer.bias)

                    exp_avg_b = ln_state['exp_avg_b']
                    exp_avg_sq_b = ln_state['exp_avg_sq_b']

                    exp_avg_b.mul_(beta1).add_(b_grad, alpha=1 - beta1)
                    exp_avg_sq_b.mul_(beta2).addcmul_(b_grad, b_grad, value=1 - beta2)

                    denom_b = exp_avg_sq_b.sqrt().add_(eps)
                    layer.bias.data.addcdiv_(exp_avg_b, denom_b, value=-step_size)

                    # Weight decay for LN bias (often we do NOT apply WD to LN bias, but is optional)
                    if weight_decay != 0.0:
                        layer.bias.data.mul_(1 - lr * weight_decay)

            elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if layer == self.first_layer:
                    A_inv = self.input_cov_inv
                else:
                    if b_updated:
                        exp_avg = state['exp_avg'].div(bias_correction)
                        sq_norm = torch.linalg.norm(exp_avg).pow(2)

                        if 'A_inv' not in state:
                            state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype)
                        else:
                            state['A_inv'].copy_(torch.eye(exp_avg.size(0), device=exp_avg.device))

                        state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))
                        state['A_inv'].div_(damping)

                    A_inv = state['A_inv']

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
