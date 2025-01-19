import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn

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


def sgd_step(optimizer):
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        step_size = group['lr']

        for p in group['params']:
            if p.grad is None:
                continue

            d_p = p.grad.data
            if weight_decay > 0:
                # Decoupled weight decay
                p.data.mul_(1.0 - step_size * weight_decay)

            p.data.add_(d_p, alpha=-step_size)


def momentum_step(optimizer):
    # update parameters
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        step_size = group['lr']
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue

            d_p = p.grad.data

            #if weight_decay != 0:
            #    d_p.add_(p.data, alpha=weight_decay)

            if momentum != 0:
                param_state = optimizer.state[p]

                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p)

                d_p = param_state['momentum_buffer'].mul_(momentum).add_(d_p)
            
            if weight_decay != 0:
                p.data.mul_(1-step_size*weight_decay)
            p.data.add_(d_p, alpha=-step_size)


def nag_step(optimizer):
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        step_size = group['lr']
        momentum = group['momentum']
        
        for p in group['params']:
            if p.grad is None:
                continue

            d_p = p.grad.data

            if weight_decay != 0:
                d_p.add_(p.data, alpha=weight_decay)

            if momentum != 0:
                param_state = optimizer.state[p]
                if 'momentum_buff' not in param_state:
                    param_state['momentum_buff'] = d_p.clone()
                else:
                    buf = param_state['momentum_buff']
                    buf.mul_(momentum).add_(d_p)
                    d_p.add_(buf, alpha=momentum)

            #if weight_decay != 0:
            #    p.data.mul_(1-step_size*weight_decay)
            p.data.add_(d_p, alpha=-step_size)


def get_vector_a(a, layer):
    """Return vectorized input activation (m_a)"""
    if isinstance(layer, nn.Linear): 
        a = torch.mean(a, list(range(len(a.shape)))[0:-1])
        if layer.bias is not None:
            a = torch.cat([a, a.new(1).fill_(1)])
        return a

    elif isinstance(layer, nn.Conv2d):
        # batch averag first
        a = torch.mean(a, dim=0, keepdim=True)
        # extract patch
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        a = torch.mean(a, [0, 1, 2])
        if layer.bias is not None:
            a = torch.cat([a, a.new(1).fill_(1)])
        return a
        
    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))


def _extract_patches(x, kernel_size, stride, padding):
    """Extract patches from convolutional layer

    Args:
      x: The input feature maps.  (batch_size, in_c, h, w)
      kernel_size: the kernel size of the conv filter (tuple of two elements)
      stride: the stride of conv operation  (tuple of two elements)
      padding: number of paddings. be a tuple of two elements
    
    Returns:
      Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x

def get_vector_g(g, layer):
    """Return vectorized deviation w.r.t. the pre-activation output (m_g)"""
    if isinstance(layer, nn.Linear):
        g = torch.mean(g, list(range(len(g.shape)))[0:-1])
        return g

    elif isinstance(layer, nn.Conv2d):
        g = torch.mean(g, [0, 2, 3])
        return g

    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))

def get_factor_A(a, layer):
    """Return KF A"""
    if isinstance(layer, nn.Linear): 
        if len(a.shape) > 2:
            a = torch.mean(a, list(range(len(a.shape)))[1:-1]) # average on sequential dim (if any)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / a.size(0))

    elif isinstance(layer, nn.Conv2d):
        # extract patch
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        a = torch.mean(a, [1, 2])  # average on spatial dims
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / a.size(0))
        
    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))

def get_factor_G(g, layer):
    """Return KF G"""
    if isinstance(layer, nn.Linear):
        if len(g.shape) > 2:
            g = torch.mean(g, list(range(len(g.shape)))[1:-1]) # average on sequential dim (if any)
        return g.t() @ (g / g.size(0))

    elif isinstance(layer, nn.Conv2d):
        g = torch.mean(g, [2, 3]) # average on spatial dims
        return g.t() @ (g / g.size(0))

    else:
        raise NotImplementedError("KFAC does not support layer: ".format(layer))

def mat_inv(x):
    u = torch.linalg.cholesky(x)
    return torch.cholesky_inverse(u)