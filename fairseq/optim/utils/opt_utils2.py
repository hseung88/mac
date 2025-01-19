import torch
import torch.nn.functional as F
import math

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

            if momentum != 0:
                param_state = optimizer.state[p]
                if 'momentum_buff' not in param_state:
                    param_state['momentum_buff'] = d_p.clone()
                else:
                    buf = param_state['momentum_buff']
                    buf.mul_(momentum).add_(d_p)
                    d_p.add_(buf, alpha=momentum)

            if weight_decay != 0:
                p.data.mul_(1-step_size*weight_decay)
                p.data.add_(d_p, alpha=-step_size)
                