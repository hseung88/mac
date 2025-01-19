import torch
import torch.nn as nn
import torch.nn.functional as F

def try_contiguous(x):
    return x if x.is_contiguous() else x.contiguous()

def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] > 0 or padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))
    x = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    x = x.view(x.size(0), x.size(1), x.size(2), -1)
    return x

def update_running_stat(aa, m_aa, stat_decay):
    m_aa.mul_(stat_decay / (1 - stat_decay)).add_(aa).mul_(1 - stat_decay)

class ComputeMatGrad:

    @classmethod
    def __call__(cls, input, grad_output, layer):
        if isinstance(layer, nn.Linear):
            return cls.linear(input, grad_output, layer)
        elif isinstance(layer, nn.Conv2d):
            return cls.conv2d(input, grad_output, layer)
        else:
            raise NotImplementedError

    @staticmethod
    def linear(input, grad_output, layer):
        with torch.no_grad():
            if layer.bias is not None:
                input = torch.cat([input, input.new_ones(input.size(0), 1)], 1)
            grad = torch.bmm(grad_output.unsqueeze(2), input.unsqueeze(1))
        return grad

    @staticmethod
    def conv2d(input, grad_output, layer):
        with torch.no_grad():
            input = _extract_patches(input, layer.kernel_size, layer.stride, layer.padding)
            input = input.view(-1, input.size(-1))
            grad_output = try_contiguous(grad_output.permute(0, 2, 3, 1)).view(-1, grad_output.size(1))
            if layer.bias is not None:
                input = torch.cat([input, input.new_ones(input.size(0), 1)], 1)
            grad = torch.einsum('ab,ac->bc', grad_output, input)
        return grad

class ComputeCovA:

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            return cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            return cls.conv2d(a, layer)
        else:
            return None

    @staticmethod
    def conv2d(a, layer):
        #batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        #spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new_ones(a.size(0), 1)], 1)
        #a = a / spatial_size
        return torch.matmul(a.t(), a) / a.size(0)

    @staticmethod
    def linear(a, layer):
        #batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new_ones(a.size(0), 1)], 1)
        return torch.matmul(a.t(), a) / a.size(0)

class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            return cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            return cls.linear(g, layer, batch_averaged)
        else:
            return None

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        #spatial_size = g.size(2) * g.size(3)
        #batch_size = g.size(0)
        g = try_contiguous(g.permute(0, 2, 3, 1)).view(-1, g.size(1))
        #if batch_averaged:
        #    g = g * batch_size
        #g = g * spatial_size
        return torch.matmul(g.t(), g) / g.size(0)

    @staticmethod
    def linear(g, layer, batch_averaged):
        g = try_contiguous(g)
        if batch_averaged:
            return torch.matmul(g.t(), g) * g.size(0)
        else:
            return torch.matmul(g.t(), g) / g.size(0)

