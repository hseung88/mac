import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def unsqueeze_and_copy(tensor, batch_size):
    expand_size = [batch_size] + [-1] * tensor.ndim
    # tensor_copi = torch.tensor(tensor.expand(expand_size),
    #                            dtype=tensor.dtype,
    #                            device=tensor.device,
    #                            requires_grad=True)
    tensor_copy = tensor.expand(expand_size).clone().detach().requires_grad_(True)

    return tensor_copy


def get_standardized_weight(weight, gain=None, eps=1e-4):
    # Get Scaled WS weight OIHW;
    fan_in = np.prod(weight.shape[-3:])
    mean = torch.mean(weight, axis=[-3, -2, -1], keepdims=True)
    var = torch.var(weight, axis=[-3, -2, -1], keepdims=True)
    weight = (weight - mean) / (var * fan_in + eps) ** 0.5
    if gain is not None:
        weight = weight * gain
    return weight


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class WSConv2d(nn.Conv2d):
    """2D Convolution with Scaled Weight Standardization and affine gain+bias."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, groups=groups, bias=bias, padding_mode=padding_mode)

        self.gain = nn.Parameter(torch.ones(self.weight.shape[0], dtype=self.weight.dtype))

    def standardize_weight(self, weight, eps=1e-4):
        """Apply scaled WS with affine gain."""
        mean = weight.mean(dim=(1, 2, 3), keepdims=True)
        var = weight.var(dim=(1, 2, 3), correction=0, keepdims=True)
        fan_in = np.prod(weight.shape[1:])

        # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var).
        eps = torch.full(var.size(), eps, device=var.device)
        scale = torch.rsqrt(torch.maximum(var * fan_in, eps)) * self.gain.view(-1, 1, 1, 1)
        shift = mean * scale

        return weight * scale - shift

    def forward(self, inputs, eps=1e-4):
        weight = self.standardize_weight(self.weight, eps)
        # weight = get_standardized_weight(self.weight, self.gain.view(-1, 1, 1, 1), eps)

        return F.conv2d(inputs, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


conv_class = {
    'conv2d': nn.Conv2d,
    'wsconv': WSConv2d,
}
