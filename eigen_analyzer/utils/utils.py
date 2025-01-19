import math
import os
import re
from glob import glob
from collections import OrderedDict
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import dill as pickle
import torch
from rich.console import Console
from rich.table import Table
from .rich_utils import print_config_tree
from common.path import path_join, makedirs


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def is_namedtuple(obj):
    return (isinstance(obj, tuple)
            and hasattr(obj, "_asdict")
            and hasattr(obj, "_fields"))


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def log_hyperparameters(cfg: DictConfig, model) -> None:
    """
    Controls which config parts are saved by lightning loggers.
    Saves additionally:
    - Number of model parameters

    Args:
        object_dict (dict): Dict object with all parameters.
    """
    cfg.network.nparams = sum(p.numel() for p in model.parameters())

    if cfg.save and cfg.trainer.create_working_dir:
        OmegaConf.save(cfg, 'config.yaml')

    if cfg.verbose:
        print_config_tree(cfg, resolve=True)


def load_config(path):
    cfg = OmegaConf.load(path)
    return cfg


def _batch_to_device(batch, device):
    return tuple(data.to(device) for data in batch)


def print_metrics(metrics, show_header=False):
    console = Console()
    table = Table(box=None)
    table.show_header = show_header
    table.show_lines = show_header
    table.show_footer = False
    table.header_style = 'bold cyan' if show_header else None

    for metric in metrics:
        if metric == 'epoch':
            table.add_column("epoch", justify='right', style='bold yellow')
        elif metric == 'step':
            table.add_column("step", justify='right', style='magenta')
        else:
            table.add_column(metric, justify='right')

    row = []
    for metric, val in metrics.items():
        if metric in ('epoch', 'step'):
            row.append(f"{val:5d}")
        else:
            row.append(f"{val:9.5f}")

    table.add_row(*row)
    console.print(table)


@torch.no_grad()
def copy_parameters(model1, model2):
    """
    Copy the corressponding parameters between two models
    """
    model1_params = OrderedDict(model1.named_parameters())
    model2_params = OrderedDict(model2.named_parameters())

    # check if both model contains the same set of keys
    assert model1_params.keys() == model2_params.keys()

    for name, param in model1_params.items():
        model2_params[name].data.copy_(param.data.detach().clone())

    model1_buffers = OrderedDict(model1.named_buffers())
    model2_buffers = OrderedDict(model2.named_buffers())

    # check if both model contains the same set of keys
    assert model1_buffers.keys() == model2_buffers.keys()

    for name, buffer in model1_buffers.items():
        # buffers are copied
        model2_buffers[name].copy_(buffer)


# ----------------------------#
#     Number calculation      #
# ----------------------------#
PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]
UNKNOWN_SIZE = "?"
unit_list = list(zip(["bytes", "kB", "MB", "GB", "TB", "PB"],
                     [0, 0, 1, 2, 2, 2]))


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


def fsize_format(num):
    """
    Human readable file size.
    copied from
    http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    """
    if num == 0:
        return "0 bytes"
    if num == 1:
        return "1 byte"

    exponent = min(int(math.log(num, 1024)), len(unit_list) - 1)
    quotient = float(num) / 1024 ** exponent
    unit, num_decimals = unit_list[exponent]
    format_string = "{:.%sf} {}" % num_decimals
    return format_string.format(quotient, unit)


# ----------------------------#
#       Serialization         #
# ----------------------------#
def save_numpy(data_array, save_path, filename):
    makedirs(save_path, warn_if_exists=False)

    with open(path_join(save_path, filename), 'wb') as fout:
        pickle.dump(data_array, fout)


def load_numpy(load_path, filename=None):
    """
    Args:
    - `load_path`: path to the directory containing the target file
    - `filename`: name of file. If None, `load_path` is used to find the file.
    """
    path_to_file = load_path if filename is None else path_join(load_path, filename)
    fpath = Path(path_to_file)

    if not fpath.is_file():
        return None

    data_array = None
    with fpath.open(mode='rb') as fin:
        data_array = pickle.load(fin)

    return data_array


def read_checkpoints(path):
    """
    Args:
    - path: string, path to the folder containing checkpoint files
    """
    chkpts = []
    for subdir in os.listdir(path):
        if os.path.isfile(os.path.join(path, subdir)):
            continue

        tmps = []
        for chkpt_file in glob(f"{os.path.join(path, subdir, 'checkpoints')}/checkpoint_*.pt"):
            p = re.compile(r"_E(\d+)S(\d+).pt")
            match = p.search(chkpt_file)
            chkpt = {
                'epoch': int(match.group(1)),
                'step': int(match.group(2)),
                'filepath': chkpt_file
            }
            tmps.append(chkpt)
        tmps.sort(key=lambda item: item.get('step'))
        chkpts.append({'subdir': subdir, 'chkpts': tmps})

    return chkpts
