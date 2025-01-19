import torch
import torch.nn as nn
from .tensor_utils import extract_patches


class AdaActStats:
    @staticmethod
    def update_linear(module, actv):
        #batch_size = actv.size(0)
        actv = actv.view(-1, actv.size(-1))
        # augment the activations if the layer has a bias term
        if module.bias is not None:
            actv = torch.cat([actv, actv.new_ones((actv.size(0), 1))], 1)
        
        A = torch.mean(actv.pow(2), axis=0)
        #A = torch.mean(actv, axis=0).pow(2)
        return A

    @staticmethod
    def update_conv(module, actv):
        #batch_size = actv.size(0)
        # a.shape = B x out_H x out_W x (in_C * kernel_H * kernel_W)
        a = extract_patches(actv, module.kernel_size, module.stride, module.padding)
        spatial_size = a.size(1) * a.size(2)  # Height x Width
        a = a.view(-1, a.size(-1))

        if module.bias is not None:
            a = torch.cat([a, a.new_ones((a.size(0), 1))], 1)

        A = torch.mean(a.pow(2), axis=0) 
        #A = torch.mean(a, axis=0).pow(2)
        return A

    STAT_UPDATE_FUNC = {
        nn.Linear: update_linear.__func__,
        nn.Conv2d: update_conv.__func__
    }

    @classmethod
    def __call__(cls, module, actv):
        return cls.STAT_UPDATE_FUNC[type(module)](module, actv)
