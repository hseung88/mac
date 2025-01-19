import torch
import torch.nn as nn
import torch.nn.functional as F


pooling_layers = {
    'avg': nn.AvgPool2d,
    'max': nn.MaxPool2d
}


activation_fn = {
    # Regular activations.
    'identity': lambda x: x,
    'tanh': torch.tanh,
    'celu': F.celu,
    'elu': F.elu,
    'gelu': F.gelu,
    'glu': F.glu,
    'leaky_relu': F.leaky_relu,
    'log_sigmoid': F.logsigmoid,
    'log_softmax': F.log_softmax,
    'relu': F.relu,
    'relu6': F.relu6,
    'selu': F.selu,
    'sigmoid': torch.sigmoid,
    'silu': F.silu,
    'swish': F.silu,
    'soft_sign': F.softsign,
    'softplus': F.softplus,
}
