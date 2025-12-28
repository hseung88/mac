from collections import Counter
import math
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
        self.layer_map = build_layer_map(model, fwd_hook_fn=self._capture_activation,
                                         supported_layers=(nn.Linear, nn.Conv2d))

    """
    def _configure(self, train_loader, net, device, cache_first_layer=False):
        # Handle the case when the model is wrapped in DistributedDataParallel
        if hasattr(net, 'module'):
            net = net.module

        # Directly capture the first layer (patch embedding) of ViTs
        first_layer = net.transformer.wte
        vocab_size = first_layer.weight.size(0)

        self.first_layer = first_layer
        self.model = net

        if cache_first_layer:
            token_count = None

            data_len = 0
            for batch in train_loader():
                if token_count is None:
                    token_count = Counter(batch)
                else:
                    token_count.update(batch)
                data_len += len(batch)

            inv_counts = torch.tensor([1 / (token_count[i] + 1.0) for i in range(vocab_size)],
                                      dtype=torch.float32).to(device)
            input_cov_inv = torch.diag_embed(inv_counts)

            self.input_cov_inv = input_cov_inv
            self.layer_map[first_layer]['fwd_hook'].remove()
    """

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

        actv = forward_input[0].detach().clone()
        attn_qkv = ('attn.c_attn' in self.layer_map[module]['name'])

        if isinstance(module, nn.Linear):
            if actv.ndim > 2:  # Expect shape [B, N, D] for transformer inputs.
                if attn_qkv:
                    B, N, D = actv.shape
                actv = actv.view(-1, actv.size(-1))

        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        if isinstance(module, nn.Embedding):
            vocab_size = self.model.config.vocab_size
            actv_flat = actv.flatten()
            token_count = Counter(actv_flat.tolist())
            counts = torch.tensor([token_count[i] for i in range(vocab_size)],
                                  dtype=torch.float32).to(actv.device)
            avg_actv = counts / len(actv_flat)
        else:
            avg_actv = actv.mean(dim=0)

        state = self.state[module]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv, device=avg_actv.device)
        state['exp_avg'].mul_(stat_decay).add_(avg_actv, alpha=1 - stat_decay)

        if attn_qkv:
            actv_b_avg = actv.view(B, N, actv.size(-1)).mean(dim=0)  # shape: [N, input_dim]

            qkv_out = _forward_output.detach().clone()
            # _forward_output is assumed to be [B, N, 3 * dim]
            B, N, three_dim = qkv_out.shape

            embed_dim = self.model.config.n_embd
            num_heads = self.model.config.n_head
            head_dim = embed_dim // num_heads

            # Reshape and permute to get q, k, v separated.
            qkv = qkv_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)  # Each is [B, num_heads, N, head_dim]

            scale = 1.0 / math.sqrt(head_dim)
            R = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
            attn = torch.softmax(R, dim=-1)
            avg_attn = attn.mean(dim=(0, 1, 2))  # [N, ]

            v_input = actv_b_avg.t() @ avg_attn

            state = self.state[module]
            if 'exp_avg_v' not in state:
                state['exp_avg_v'] = torch.zeros_like(v_input, device=v_input.device)
            state['exp_avg_v'].mul_(stat_decay).add_(v_input, alpha=1 - stat_decay)

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
        if (self._step % self.Tcov) == 0:
            self.emastep += 1
        if (self._step % self.Tinv) == 0:
            b_updated = True

        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                # if isinstance(layer, nn.Embedding):
                #     A_inv = torch.diag_embed(1.0 / state['exp_avg'])
                # else:
                if b_updated:
                    bias_correction = 1.0 - (stat_decay ** self.emastep)
                    exp_avg = state['exp_avg'].div(bias_correction).to(grad_mat.dtype)
                    sq_norm = torch.dot(exp_avg, exp_avg)

                    if 'A_inv' not in state:
                        state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype)
                    else:
                        state['A_inv'].copy_(torch.eye(exp_avg.size(0), device=exp_avg.device, dtype=exp_avg.dtype))

                    state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))

                A_inv = state['A_inv'].to(grad_mat.dtype)

                # if layer != self.first_layer and 'attn.qkv' in self.layer_map[layer]['name']:
                if 'attn.qkv' in self.layer_map[layer]['name']:
                    # embed_dim = self.model.embed_dim
                    dim = grad_mat.size(0) // 3

                    # Split grad_mat into q, k, and v parts
                    # q_grad_full = grad_mat[:embed_dim, :]  # shape: [embed_dim, input_dim]
                    # k_grad_full = grad_mat[embed_dim:2 * embed_dim, :]  # shape: [embed_dim, input_dim]
                    # v_grad_full = grad_mat[2 * embed_dim:, :]  # shape: [embed_dim, input_dim]
                    q_grad_full = grad_mat[:dim, :]
                    k_grad_full = grad_mat[dim:2 * dim, :]
                    v_grad_full = grad_mat[2 * dim:3 * dim, :]

                    if b_updated:
                        bias_correction = 1.0 - (stat_decay ** self.emastep)
                        # Update per-head inverse preconditioners
                        exp_avg_v = state['exp_avg_v'].div(bias_correction).to(grad_mat.dtype)  # [num_heads, input_dim]
                        sq_norm_v = torch.dot(exp_avg_v, exp_avg_v)

                        if 'V_inv' not in state:
                            state['V_inv'] = torch.eye(exp_avg_v.size(0), device=exp_avg_v.device, dtype=exp_avg_v.dtype)
                        else:
                            state['V_inv'].copy_(torch.eye(exp_avg_v.size(0), device=exp_avg_v.device, dtype=exp_avg_v.dtype))

                        state['V_inv'].sub_(torch.outer(exp_avg_v, exp_avg_v).div_(damping + sq_norm_v))

                    V_inv = state['V_inv'].to(grad_mat.dtype)

                    q_grad_precond = q_grad_full @ A_inv
                    k_grad_precond = k_grad_full @ A_inv
                    v_grad_precond = v_grad_full @ V_inv

                    new_grad = torch.cat([q_grad_precond, k_grad_precond, v_grad_precond], dim=0)
                    v = new_grad
                else:
                    v = grad_mat @ A_inv

                if hasattr(layer, 'bias') and layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        # momentum_step(self)
        nag_step(self)
        self._step += 1

        return loss
