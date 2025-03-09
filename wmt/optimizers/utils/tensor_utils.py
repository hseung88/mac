import torch
import torch.nn.functional as F


def moving_average(new_val, stat, decay):
    stat.mul_(decay).add_(new_val, alpha=1.0 - decay)


def extract_patches(x, kernel_size, stride, padding):
    """
    x: input feature map of shape (B x C x H x W)
    kernel_size: the kernel size of the conv filter (tuple of two elements)
    stride: the stride of conv operation  (tuple of two elements)
    padding: number of paddings. be a tuple of two elements

    return: (batch_size, out_h, out_w, in_c*kh*kw)
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
