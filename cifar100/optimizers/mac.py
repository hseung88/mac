import math
from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, momentum_step


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
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model, fwd_hook_fn=self._capture_activation)

    def _configure(self, train_loader, net, device):
        n_batches = len(train_loader)
        cov_mat = None

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

                # A = torch.einsum('ij,jk->ik', actv.t(), actv) / actv.size(0)  # Optimized matrix multiplication
                A = torch.matmul(actv.t(), actv) / actv.size(0)
                if cov_mat is None:
                    cov_mat = A
                else:
                    cov_mat.add_(A)

            cov_mat /= n_batches

        self.first_layer = first_layer
        eye_matrix = torch.eye(cov_mat.size(0), device=device, dtype=cov_mat.dtype)
        self.input_cov_inv = torch.linalg.inv(cov_mat + self.damping * eye_matrix)
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

        if (self._step % self.Tcov) != 0:
            return

        group = self.param_groups[0]
        stat_decay = group['stat_decay']

        # If the current module is the qkv layer, extract q and k from _forward_output
        if 'attn.qkv' in self.layer_map[module]['name']:
            # _forward_output is a tensor of shape [B, N, 3 * dim].
            B, N, three_dim = _forward_output.shape
            # The attention module typically has num_heads and head_dim attributes
            num_heads = self.model.deit.num_heads
            head_dim = self.model.deit.head_dim
            # Reshape and permute to separate the qkv tensor.
            qkv = _forward_output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)  # q, k shape: [B, num_heads, N, head_dim]

            # Compute the mean over the batch and token dimensions for each head.
            avg_q = q.mean(dim=(0, 1, 2))  # shape: [1, head_dim]
            avg_k = k.mean(dim=(0, 1, 2))  # shape: [1, head_dim]

            state = self.state[module]
            if 'exp_avg_q' not in state:
                state['exp_avg_q'] = torch.zeros_like(avg_q, device=avg_q.device)
                state['exp_avg_k'] = torch.zeros_like(avg_k, device=avg_k.device)
            state['exp_avg_q'].mul_(stat_decay).add_(avg_q, alpha=1 - stat_decay)
            state['exp_avg_k'].mul_(stat_decay).add_(avg_k, alpha=1 - stat_decay)

        actv = forward_input[0].data
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:  # for Linear layers in transformers
                actv = actv.view(-1, actv.size(-1))
        elif isinstance(module, nn.LayerNorm):
            if actv.ndim > 2:
                actv = actv.view(-1, actv.size(-1))
            # Standardize inputs to mimic LayerNorm normalization.
            mean = actv.mean(dim=-1, keepdim=True)
            var = actv.var(dim=-1, unbiased=False, keepdim=True)
            actv = (actv - mean) / torch.sqrt(var + module.eps)

        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(dim=0)

        state = self.state[module]
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
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.LayerNorm)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if layer == self.first_layer:
                    A_inv = self.input_cov_inv.to(grad_mat.dtype)
                elif 'attn.qkv' in self.layer_map[layer]['name']:
                    three_d, input_dim = grad_mat.shape
                    d = three_d // 3

                    q_grad = grad_mat[:d, :]  # shape: [d, input_dim]
                    k_grad = grad_mat[d:2 * d, :]  # shape: [d, input_dim]
                    v_grad = grad_mat[2 * d:, :]  # shape: [d, input_dim]

                    if b_updated:
                        bias_correction = 1.0 - (stat_decay ** self.emastep)
                        exp_avg_q = state['exp_avg_q'].div(bias_correction).view(-1).to(grad_mat.dtype)
                        exp_avg_k = state['exp_avg_k'].div(bias_correction).view(-1).to(grad_mat.dtype)
                        sq_norm_q = torch.linalg.norm(exp_avg_q).pow(2)
                        sq_norm_k = torch.linalg.norm(exp_avg_k).pow(2)

                        if 'q_inv' not in state:
                            state['q_inv'] = torch.eye(exp_avg_q.size(0), device=exp_avg_q.device, dtype=exp_avg_q.dtype)
                            state['k_inv'] = torch.eye(exp_avg_k.size(0), device=exp_avg_k.device, dtype=exp_avg_k.dtype)
                        else:
                            state['q_inv'].copy_(torch.eye(exp_avg_q.size(0), device=exp_avg_q.device, dtype=exp_avg_q.dtype))
                            state['k_inv'].copy_(torch.eye(exp_avg_k.size(0), device=exp_avg_k.device, dtype=exp_avg_k.dtype))

                        state['q_inv'].sub_(torch.outer(exp_avg_q, exp_avg_q).div_(damping + sq_norm_q))
                        state['k_inv'].sub_(torch.outer(exp_avg_k, exp_avg_k).div_(damping + sq_norm_k))

                    q_inv = state['q_inv'].to(grad_mat.dtype)
                    k_inv = state['q_inv'].to(grad_mat.dtype)

                    q_precond = k_inv @ q_grad
                    k_precond = q_inv @ k_grad
                    # For v, we leave the gradient unchanged
                    new_grad = torch.cat([q_precond, k_precond, v_grad], dim=0)
                    grad_mat = new_grad
                else:
                    if b_updated:
                        bias_correction = 1.0 - (stat_decay ** self.emastep)
                        exp_avg = state['exp_avg'].div(bias_correction).to(grad_mat.dtype)
                        sq_norm = torch.linalg.norm(exp_avg).pow(2)

                        if 'A_inv' not in state:
                            state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype)
                        else:
                            state['A_inv'].copy_(torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype))

                        state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))
                        #state['A_inv'].div_(damping)

                    A_inv = state['A_inv'].to(grad_mat.dtype)

            if isinstance(layer, nn.LayerNorm):
                v = A_inv @ grad_mat
                layer.weight.grad.data.copy_(v.view_as(layer.weight))
            else:
                v = grad_mat @ A_inv
                if layer.bias is not None:
                    v_weight = v[:, :-1]
                    v_bias = v[:, -1:]
                    layer.weight.grad.data.copy_(v_weight.view_as(layer.weight))
                    layer.bias.grad.data.copy_(v_bias.view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        momentum_step(self)
        self._step += 1

        return loss
