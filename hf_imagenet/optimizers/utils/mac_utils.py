import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
from typing import Iterable
import functools


def extract_patches(x, kernel_size, stride, padding, depthwise=False):
    """
    x: input feature map of shape (B x C x H x W)
    kernel_size: the kernel size of the conv filter (tuple of two elements)
    stride: the stride of conv operation  (tuple of two elements)
    padding: number of paddings. be a tuple of two elements

    return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if isinstance(padding, str):
        padding = (
            ((stride[0] - 1) * x.size(2) - stride[0] + kernel_size[0]) // 2,
            ((stride[1] - 1) * x.size(3) - stride[1] + kernel_size[1]) // 2,
        )

    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims

    # Tensor.unfold(dimension, size, step)
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    # x.shape = [B x C x oH x oW x K x K]

    if depthwise:
        x = x.reshape(x.size(0) * x.size(1) * x.size(2) * x.size(3),
                      x.size(4) * x.size(5))
    else:
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        # x.shape = [B x oH x oW x (C x K x K)]
        x = x.view(
            x.size(0) * x.size(1) * x.size(2),
            x.size(3) * x.size(4) * x.size(5))

    return x


def reshape_grad(layer):
    """
    returns the gradient reshaped for KFAC, shape=[batch_size, output_dim, input_dim]
    """
    classname = layer.__class__.__name__

    g = layer.weight.grad

    if classname == 'Conv2d':
        grad_mat = g.view(g.size(0), -1)  # n_filters * (in_c * kw * kh)
    else:
        grad_mat = g

    # include the bias into the weight
    if layer.bias is not None:
        grad_mat = torch.cat([grad_mat, layer.bias.grad.view(-1, 1)], 1)

    return grad_mat


def parametrized_modules(module: nn.Module) -> Iterable[nn.Module]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters (as opposed to "wrapper modules" that just organize modules).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in module.named_modules()
        if any(p is not None for p in m.parameters(recurse=False))
    )


def trainable_modules(module: nn.Module) -> Iterable[nn.Module]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters and are trainable (ie they want a grad).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in parametrized_modules(module)
        if any(p.requires_grad for p in m.parameters(recurse=False))
    )


def grad_layers(module, memo=None, prefix=''):
    if memo is None:
        memo = set()

    if module not in memo:
        memo.add(module)

        if bool(module._modules):
            for name, module in module._modules.items():
                if module is None:
                    continue
                sub_prefix = prefix + ('.' if prefix else '') + name
                for ll in grad_layers(module, memo, sub_prefix):
                    yield ll
        else:
            if bool(module._parameters):
                grad_param = []

                for pname, param in module._parameters.items():
                    if param is None:
                        continue

                    if param.requires_grad:
                        grad_param.append((pname, param))

                if grad_param:
                    yield module, prefix, grad_param


def build_layer_map(model, fwd_hook_fn=None, bwd_hook_fn=None,
                    supported_layers=(nn.Linear, nn.Conv2d)):
    layer_map = {}

    for layer, prefix, params in grad_layers(model):
        if isinstance(layer, supported_layers):
            h_fwd_hook = layer.register_forward_hook(fwd_hook_fn) if fwd_hook_fn else None
            h_bwd_hook = layer.register_full_backward_hook(bwd_hook_fn) if bwd_hook_fn else None
        else:
            h_fwd_hook = None
            h_bwd_hook = None

        layer_map[layer] = {
            'name': prefix,
            'params': params,  # list of tuples; each tuple is of form: (pname, parameter)
            'fwd_hook': h_fwd_hook,
            'bwd_hook': h_bwd_hook
        }
    return layer_map


def sgd_step(optimizer):
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        step_size = group['lr']

        for p in group['params']:
            if p.grad is None:
                continue

            d_p = p.grad.data
            d_p.add_(p.data, alpha=weight_decay)

            #p.data.mul_(1.0 - step_size * weight_decay)
            p.data.add_(d_p, alpha=-step_size)


def momentum_step(optimizer):
    # update parameters for MAC layers only (not AdamW-managed ones)
    adamw_params = set()
    if hasattr(optimizer, 'adamw_param_groups'):
        for group in optimizer.adamw_param_groups:
            for p in group['params']:
                adamw_params.add(p)

    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        step_size = group['lr']
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None or p in adamw_params:
                continue

            d_p = p.grad.data
            param_state = optimizer.state[p]

            if 'momentum_buffer' not in param_state:
                param_state['momentum_buffer'] = torch.zeros_like(p)
            d_p = param_state['momentum_buffer'].mul_(momentum).add_(d_p)

            p.data.mul_(1 - step_size * weight_decay)
            p.data.add_(d_p, alpha=-step_size)


def sign_gd_step(optimizer):
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        step_size = group['lr']
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue

            d_p = p.grad.data

            param_state = optimizer.state[p]
            if 'momentum_buffer' not in param_state:
                param_state['momentum_buffer'] = torch.zeros_like(p)
            d_p = param_state['momentum_buffer'].mul_(momentum).add_(d_p)

            p.data.mul_(1 - step_size * weight_decay)
            p.data.add_(torch.sign(d_p), alpha=-step_size)


def nag_step(optimizer):
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        step_size = group['lr']
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue

            d_p = p.grad.data
            d_p.add_(p.data, alpha=weight_decay)

            param_state = optimizer.state[p]
            if 'momentum_buff' not in param_state:
                param_state['momentum_buff'] = d_p.clone()
            else:
                buf = param_state['momentum_buff']
                buf.mul_(momentum).add_(d_p)
                d_p.add_(buf, alpha=momentum)

            #p.data.mul_(1 - step_size * weight_decay)
            p.data.add_(d_p, alpha=-step_size)


def adamw_step(optimizer):
    for group in optimizer.param_groups:
        lr = group['lr']
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        weight_decay = group['weight_decay']

        for p in group['params']:
            state = optimizer.state[p]
            if p.grad is None:
                continue

            grad = p.grad

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(eps)

            bias_correction1 = 1.0 - beta1 ** state['step']
            bias_correction2 = 1.0 - beta2 ** state['step']
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            p.data.mul_(1 - lr * weight_decay)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)


def ema_step(optimizer):
    for group in optimizer.param_groups:
        lr = group['lr']
        momentum = group['momentum']
        weight_decay = group['weight_decay']

        for p in group['params']:
            state = optimizer.state[p]
            if p.grad is None:
                continue

            grad = p.grad

            if len(state) == 0:
                state['step'] = 0
                state['ema_grad'] = torch.zeros_like(p)

            exp_avg = state['ema_grad']
            state['step'] += 1
            exp_avg.mul_(momentum).add_(grad, alpha=1 - momentum)

            bias_correction1 = 1.0 - momentum ** state['step']
            step_size = lr / bias_correction1

            p.data.mul_(1 - lr * weight_decay)
            p.data.add_(exp_avg, alpha=-step_size)