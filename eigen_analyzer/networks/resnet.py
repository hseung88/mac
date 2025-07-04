"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.input_size = [6]
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.kernels = [(self.conv1.weight, self.input_size)]
        self.bns = []
        self.bn1 = nn.BatchNorm2d(planes)
        self.bns.append(self.bn1)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.kernels.append((self.conv2.weight, self.input_size))
        self.bn2 = nn.BatchNorm2d(planes)
        self.bns.append(self.bn2)

        self.shortcut = nn.Sequential()
        self.do_shortcut = stride != 1 or in_planes != self.expansion * planes
        if self.do_shortcut:
            conv = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            bn = nn.BatchNorm2d(self.expansion * planes)
            self.shortcut = nn.Sequential(conv, bn)
            self.kernels.append((conv.weight, self.input_size))
            self.bns.append(bn)

    def get_all_bns(self):
        return self.bns

    def get_all_kernels(self):
        return self.kernels

    def forward(self, x):
        self.input_size[0] = x.shape[-1]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.do_shortcut:
            # to be one 1 lip
            out = out / 2
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        self.do_shortcut = stride != 1 or in_planes != self.expansion * planes
        if self.do_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200, width=1):
        super(ResNet, self).__init__()
        self.in_planes = 16 * width

        self.conv1 = nn.Conv2d(3, 16 * width, kernel_size=3, stride=1, padding=1, bias=False)
        self.kernels = [(self.conv1.weight, [32])]
        self.bns = []
        self.bn1 = nn.BatchNorm2d(16 * width)
        self.bns.append(self.bn1)
        self.layer1 = self._make_layer(block, 16 * width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128 * width, num_blocks[3], stride=2)
        self.linear = nn.Linear(128 * width * block.expansion, num_classes)

    def get_all_bns(self):
        return self.bns

    def get_all_kernels(self):
        return self.kernels

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            one_block = block(self.in_planes, planes, stride)
            self.kernels.extend(one_block.get_all_kernels())
            self.bns.extend(one_block.get_all_bns())
            layers.append(one_block)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet9(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
