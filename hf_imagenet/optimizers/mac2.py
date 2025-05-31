import math
from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
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

        # AdamW sub-optimizer for embedding, LayerNorm
        self.adamw_layers = []
        self.adamw_param_groups = []
        self.adamw = None

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model, fwd_hook_fn=self._capture_activation)

        adamw_params = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, nn.LayerNorm)):
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        adamw_params.append(p)
                        self.adamw_layers.append(module)
        if adamw_params:
            self.adamw_param_groups = [{'params': adamw_params}]
            self.adamw = AdamW(self.adamw_param_groups, lr=0.001, weight_decay=0.05)

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
        attn_qkv = ('attn.qkv' in self.layer_map[module]['name'])

        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:
                if attn_qkv:
                    B, N, D = actv.shape
                actv = actv.view(-1, actv.size(-1))

        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(dim=0)

        state = self.state[module]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv, device=avg_actv.device)
        state['exp_avg'].mul_(stat_decay).add_(avg_actv, alpha=1 - stat_decay)

        if attn_qkv:
            actv_b_avg = actv.view(B, N, actv.size(-1)).mean(dim=0)
            qkv_out = _forward_output.detach().clone()
            B, N, three_dim = qkv_out.shape
            if hasattr(self.model, 'layers'):
                layer_name = self.layer_map[module]['name']
                match = re.search(r'layers\.(\d+)\.blocks\.(\d+)', layer_name)
                stage_idx = int(match.group(1))
                block_idx = int(match.group(2))
                num_heads = self.model.layers[stage_idx].blocks[block_idx].attn.num_heads
                head_dim = self.model.layers[stage_idx].blocks[block_idx].dim // num_heads
            elif hasattr(self.model, 'blocks'):
                num_heads = self.model.blocks[0].attn.num_heads
                head_dim = self.model.embed_dim // num_heads
            qkv = qkv_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)

            scale = 1.0 / math.sqrt(head_dim)
            R = (q @ k.transpose(-2, -1)) * scale
            attn = torch.softmax(R, dim=-1)
            avg_attn = attn.mean(dim=(0, 1, 2))

            v_input = actv_b_avg.t() @ avg_attn

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
        if self._step % self.Tinv == 0:
            b_updated = True
            self.emastep += 1

        for layer in self.layer_map:
            if layer in self.adamw_layers:
                continue

            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

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

                if 'attn.qkv' in self.layer_map[layer]['name']:
                    embed_dim = self.model.embed_dim
                    q_grad_full = grad_mat[:embed_dim, :]
                    k_grad_full = grad_mat[embed_dim:2 * embed_dim, :]
                    v_grad_full = grad_mat[2 * embed_dim:, :]

                    if b_updated:
                        bias_correction = 1.0 - (stat_decay ** self.emastep)
                        exp_avg_v = state['exp_avg_v'].div(bias_correction).to(grad_mat.dtype)
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

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        if self.adamw:
            self.adamw.step()

        momentum_step(self)
        self._step += 1

        return loss
