import torch.nn as nn

from networks.module.activations import activation_fn, pooling_layers
from utils.torch_utils import conv_output_size


class LeNet5(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 32,
        act_fn: str = 'tanh',
        pool_fn: str = 'max'
    ):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        out_size = conv_output_size(image_size, 5, 1, 0)
        out_size = conv_output_size(out_size, 2, 2, 0)
        out_size = conv_output_size(out_size, 5, 1, 0)
        out_size = conv_output_size(out_size, 2, 2, 0)

        self.L1 = nn.Linear(16*out_size*out_size, 120)  # for CIFAR10, 16x5x5
        self.L2 = nn.Linear(120, 84)
        self.L3 = nn.Linear(84, num_classes)

        self.act = activation_fn[act_fn]
        self.pool = pooling_layers[pool_fn](kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool(self.act(self.conv1(x)))
        out = self.pool(self.act(self.conv2(out)))

        out = out.view(out.size(0), -1)
        out = self.act(self.L1(out))
        out = self.act(self.L2(out))
        out = self.L3(out)

        return out
