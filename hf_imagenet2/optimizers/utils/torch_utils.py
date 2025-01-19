from typing import Iterable
import functools
import torch
import torch.nn as nn
from .utils_ import grad_layers


def no_grad_func(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return new_func


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


def build_layer_map(model, fwd_hook_fn=None, bwd_hook_fn=None,
                    supported_layers=(nn.Linear, nn.Conv2d)):
    layer_map = {}

    for layer, prefix, params in grad_layers(model):
        if isinstance(layer, supported_layers):
            h_fwd_hook = layer.register_forward_hook(fwd_hook_fn) if fwd_hook_fn else None
            h_bwd_hook = layer.register_full_backward_hook(bwd_hook_fn) if bwd_hook_fn else None
        else:
            h_fwd_hook = None
            h_bwd_hook = None

        layer_map[layer] = {
            'name': prefix,
            'params': params,  # list of tuples; each tuple is of form: (pname, parameter)
            'fwd_hook': h_fwd_hook,
            'bwd_hook': h_bwd_hook
        }
    return layer_map
