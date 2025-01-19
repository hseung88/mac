from typing import Iterable
import math
import functools
import torch
import torch.nn as nn


def no_grad_func(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return new_func


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Returns the output image size of a convolutional layer
    """
    out = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return math.floor(out / stride + 1)

# ----------------------#
#     Parameters        #
# ----------------------#


def get_parameter_count(model):
    return sum(p.numel() for p in model.parmaeters())


def requires_grad(module: nn.Module, *, recurse: bool = False) -> bool:
    """
    Checks if any parameters in a specified module require gradients.

    Args:
        module: PyTorch module whose parameters are to be examined.
        recurse: Flag specifying if the gradient requirement check should
            be applied recursively to sub-modules of the specified module

    Returns:
        Flag indicate if any parameters require gradients
    """
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def trainable_parameters(module):
    """
    Recursively iterates over all parameters, returning those that
    are trainable (i.e., they want a grad).
    """
    yield from (
        (p_name, p) for (p_name, p) in module.named_parameters() if p.requires_grad
    )


def parametrized_modules(module: nn.Module) -> Iterable[nn.Module]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters (as opposed to "wrapper modules" that just organize modules).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in module.named_modules()
        if any(p is not None for p in m.parameters(recurse=False))
    )


def trainable_modules(module: nn.Module) -> Iterable[nn.Module]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters and are trainable (ie they want a grad).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in parametrized_modules(module)
        if any(p.requires_grad for p in m.parameters(recurse=False))
    )


def grad_layers(module, memo=None, prefix=''):
    """
    returns modules having trainable parameters (i.e., those require calculating gradient)

    This function is deprecated and use `trainable_modules()` instead.
    """
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


def build_layer_map(model, fwd_hook_fn=None, bwd_hook_fn=None,
                    supported_layers=None):
    """
    Args:
    - `supported_layers`: list of supported layer classes, e.g., (nn.Linear, nn.Conv2d)
    """
    layer_map = {}

    for mod_name, module in trainable_modules(model):
        if supported_layers is None or isinstance(module, supported_layers):
            h_fwd_hook = module.register_forward_hook(fwd_hook_fn) if fwd_hook_fn else None
            h_bwd_hook = module.register_full_backward_hook(bwd_hook_fn) \
                if bwd_hook_fn else None

            layer_map[module] = {
                'name': mod_name,
                'params': list(trainable_parameters(module)),
                'fwd_hook': h_fwd_hook,
                'bwd_hook': h_bwd_hook
            }
    return layer_map


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


@torch.no_grad()
def collect_grads(model):
    """
    Returns the gradients of model parmaeters in a dictionary
    """
    grads = {
        pname: p.grad.detach().clone().cpu().numpy()
        for pname, p in trainable_parameters(model)
    }

    return grads
