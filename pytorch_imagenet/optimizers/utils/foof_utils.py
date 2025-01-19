import torch
import torch.nn as nn
from .tensor_utils import extract_patches

class FoofStats:
    @staticmethod
    def update_linear(module, actv):
        actv = actv.reshape(-1, actv.size(-1))
        if module.bias is not None:
            bias_ones = actv.new_ones((actv.size(0), 1))
            actv = torch.cat([actv, bias_ones], dim=1)
        A = torch.matmul(actv.t(), actv) / actv.size(0)
        return A

    @staticmethod
    def update_conv(module, actv):
        patches = extract_patches(actv, module.kernel_size, module.stride, module.padding)
        patches = patches.reshape(-1, patches.size(-1))
        if module.bias is not None:
            bias_ones = patches.new_ones((patches.size(0), 1))
            patches = torch.cat([patches, bias_ones], dim=1)
        A = torch.matmul(patches.t(), patches) / patches.size(0)
        return A

    STAT_UPDATE_FUNC = {
        nn.Linear: update_linear,
        nn.Conv2d: update_conv
    }

    @classmethod
    def __call__(cls, module, actv):
        update_func = cls.STAT_UPDATE_FUNC.get(type(module))
        if update_func is None:
            raise NotImplementedError(f"Module type {type(module)} is not supported.")
        return update_func(module, actv)
