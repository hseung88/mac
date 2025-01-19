import os
import subprocess
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from rich.console import Console
from rich.table import Table
import common.logging as logging
from common.path import create_working_dir, path_join, makedirs


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min
log = logging.logger


def check_gpu_memory(verbose=False):
    result = subprocess.check_output(['nvidia-smi',
                                      '--query-gpu=index,name,memory.total,memory.used,',
                                      '--format=csv,nounits,noheader'],
                                     encoding='utf-8')

    devices = [x.split(',') for x in result.strip().split('\n')]

    if verbose:
        table = Table(title="GPU Device(s)")

        table.add_column("Id", justify="center", style="cyan")
        table.add_column("Name", justify="left")
        table.add_column("Memory (total)", justify="right")
        table.add_column("Memory (used)", justify="right", style="yellow")

        for device in devices:
            table.add_row(*device)

        console = Console()
        console.print(table)

    gpu_memory = [int(device[3]) for device in devices]

    return gpu_memory


def cuda_setup(gpu_id=-1, deterministic=False, verbose=False):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        if gpu_id < 0:
            memory_usage = check_gpu_memory(verbose=verbose)
            gpu_id = np.argmin(memory_usage)
            log.info(f"GPU:{gpu_id} is selected.")

        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = torch.device("cpu")

    return device


def initialize(cfg):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.app_root = get_original_cwd()
    if cfg.save and cfg.trainer.create_working_dir:
        choices = HydraConfig.get().runtime.choices
        wdir_name = f"{cfg.dataset.dname}_{cfg.task.name}_{cfg.network.name}_{choices['optimizer']}"
        wpostfix = cfg.get('wdir_postfix', None)
        if wpostfix is not None:
            wdir_name += f"_{wpostfix}"

        exp_name = cfg.get('exp_name', None)

        if exp_name is not None and exp_name.strip() == '':
            exp_name = None
        wdir = create_working_dir(path_join('outputs', wdir_name), exp_name)
        makedirs(path_join(wdir, 'checkpoints'), warn_if_exists=True)

        logging.get_logger(file_name='main.log', level='DEBUG')
        cfg.return_dir = get_original_cwd()
        cfg.working_dir = wdir
    else:
        cfg.working_dir = get_original_cwd()

    if cfg.verbose:
        level = "DEBUG" if cfg.debug else "INFO"
        logging.get_logger(level=level)

    deterministic = cfg.get("deterministic", False)
    gpu_id = cfg.get('gpu_id', -1)

    if deterministic:
        SEED = cfg.get('seed', 1234)
        seed_everything(SEED)

    device = cuda_setup(gpu_id=gpu_id, deterministic=deterministic, verbose=cfg.verbose)
    # Then in training code before the train loop
    set_debug_apis(state=False)

    return device


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


def seed_everything(SEED=None):
    if SEED is None:
        SEED = random.randint(min_seed_value, max_seed_value)
    elif not isinstance(SEED, int):
        SEED = int(SEED)

    log.debug(f"Setting the global seed to {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    return SEED
