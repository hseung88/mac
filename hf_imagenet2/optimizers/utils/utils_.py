import math
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Mapping, Sequence
import torch


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def is_namedtuple(obj):
    return (isinstance(obj, tuple)
            and hasattr(obj, "_asdict")
            and hasattr(obj, "_fields"))


def apply_to_collection(data, dtype, func, *args, **kwargs):
    if isinstance(data, dtype):
        return func(data, *args, **kwargs)

    elem_type = type(data)

    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = apply_to_collection(v, dtype, func, *args, **kwargs)
            out.append((k, v))

        if isinstance(data, defaultdict):
            return elem_type(data.decault_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple_ = is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple_ or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(d, dtype, func, *args, **kwargs)
            out.append(v)
    return elem_type(*out) if is_namedtuple_ else elem_type(out)


# ------------------------ #
#        Model             #
# ------------------------ #


def get_parameter_count(model):
    param_count = 0
    for param in model.parameters():
        param_count += torch.numel(param)

    return param_count


def grad_layers(module, memo=None, prefix=''):
    if memo is None:
        memo = set()

    if module not in memo:
        memo.add(module)

        if bool(module._modules):
            for name, module in module._modules.items():
                if module is None:
                    continue
                sub_prefix = prefix + ('.' if prefix else '') + name
                for ll in grad_layers(module, memo, sub_prefix):
                    yield ll
        else:
            if bool(module._parameters):
                grad_param = []

                for pname, param in module._parameters.items():
                    if param is None:
                        continue

                    if param.requires_grad:
                        grad_param.append((pname, param))

                if grad_param:
                    yield module, prefix, grad_param


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def model_params(layer_map, per_layer=False):
    """
    Returns a list of parameters to clip
    """

    for layer, layer_info in layer_map.items():
        parameters = [p for _, p in layer_info['params']]

        if per_layer:
            yield parameters
        else:
            for param in parameters:
                yield param


def layer_params(layer_map):
    for layer, layer_info in layer_map.items():
        for pname, p in layer_info['params']:
            yield (layer, p, pname)


# https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/7
def conv2d_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(padding), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)

    return h, w


# ----------------------------#
#     Number calculation      #
# ----------------------------#
PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]
UNKNOWN_SIZE = "?"


# copied from
# https://github.com/Lightning-AI/lightning/blob
#      /511a070c529144a76ec6c891a0e2c75ddaec8e77
#      /src/pytorch_lightning/utilities/model_summary/model_summary.py
def get_human_readable_count(number: int) -> str:
    """Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    # don't abbreviate beyond trillions
    num_groups = min(num_groups, len(labels))
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"

    return f"{number:,.1f} {labels[index]}"
