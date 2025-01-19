"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .custom_layers import conv_class


layer_config = {
    8: [1, 1, 1],
    14: [2, 2, 2],
    20: [3, 3, 3],
    32: [5, 5, 5],
    44: [7, 7, 7],
    56: [9, 9, 9],
    110: [18, 18, 18],
    1202: [200, 200, 200]
}


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d,
                 option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0, planes//4, planes//4),
                                                  "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    conv_layer(in_planes, self.expansion * planes, kernel_size=1,
                               stride=stride, bias=False),
                    norm_layer(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 20,
        which_conv: str = 'conv2d',  # {'wsconv', 'scaledstdconv', 'conv2d' }
        which_norm: str = 'BN',
        groups: int = 16,
        affine: bool = True,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 16

        block = BasicBlock
        num_blocks = layer_config[depth]
        conv_layer = conv_class[which_conv]
        if which_norm == 'BN':
            norm_layer = partial(nn.BatchNorm2d, affine=affine)
        elif which_norm == 'GN':
            norm_layer = partial(nn.GroupNorm, groups, affine=affine)
        else:
            raise ValueError(f'Unknown normalization type: {which_norm}')

        self.conv1 = conv_layer(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1,
                                       conv_layer=conv_layer, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2,
                                       conv_layer=conv_layer, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,
                                       conv_layer=conv_layer, norm_layer=norm_layer)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride,
                    conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, conv_layer, norm_layer))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
