import math
from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, momentum_step
from . import FairseqOptimizer, register_optimizer


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


log.basicConfig(level=log.DEBUG)  # Set logging level to debug for detailed info


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
            if actv.ndim > 2:  # Expect shape [B, N, d] for transformer inputs.
                if attn_qkv:
                    B, N, D = actv.shape
                actv = actv.view(-1, actv.size(-1))

        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(dim=0)

        # Use id(module) as the unique layer key
        layer_key = f"{id(module)}"
        state = self.state[layer_key]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv, device=avg_actv.device)
        state['exp_avg'].mul_(stat_decay).add_(avg_actv, alpha=1 - stat_decay)

        if attn_qkv:
            actv_b_avg = actv.view(B, N, actv.size(-1)).mean(dim=0)  # shape: [N, input_dim]

            qkv_out = _forward_output.detach().clone()
            # _forward_output is assumed to be [B, N, 3 * dim]
            B, N, three_dim = qkv_out.shape
            num_heads = self.model.blocks[0].attn.num_heads
            head_dim = self.model.embed_dim // num_heads
            # Reshape and permute to get q, k, v separated.
            qkv = qkv_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)  # Each is [B, num_heads, N, head_dim]

            scale = 1.0 / math.sqrt(head_dim)
            R = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
            attn = torch.softmax(R, dim=-1)
            avg_attn = attn.mean(dim=(0, 1, 2))  # [N, ]

            v_input = actv_b_avg.t() @ avg_attn

            state = self.state[layer_key]
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
            layer_key = f"{id(layer)}"
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer_key]
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
                    #state['A_inv'].div_(damping)

                A_inv = state['A_inv'].to(grad_mat.dtype)

                if 'attn.qkv' in self.layer_map[layer]['name']:
                    embed_dim = self.model.embed_dim

                    # Split grad_mat into q, k, and v parts
                    q_grad_full = grad_mat[:embed_dim, :]  # shape: [embed_dim, input_dim]
                    k_grad_full = grad_mat[embed_dim:2 * embed_dim, :]  # shape: [embed_dim, input_dim]
                    v_grad_full = grad_mat[2 * embed_dim:, :]  # shape: [embed_dim, input_dim]

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

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        momentum_step(self)
        self._step += 1

        return loss