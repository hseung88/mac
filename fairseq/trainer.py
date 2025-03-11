#!/usr/bin/env python
"""
Train a network on a single GPU.

This file is a modified version of fairseq/fairseq/trainer.py with all distributed
settings removed. It is configured to run on one GPU (or CPU if CUDA is unavailable).
"""

import contextlib
import logging
import os
import sys
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from fairseq import checkpoint_utils, models, optim, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.models.ema import build_ema
from fairseq.nan_detector import NanDetector
from fairseq.optim import lr_scheduler

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    Main training class for single GPU training.

    All distributed training functionality is removed, so this class
    assumes a single process and a single device.
    """

    def __init__(self, cfg: FairseqConfig, task, model, criterion, quantizer=None):
        if isinstance(cfg, Namespace):
            logger.warning(
                "argparse.Namespace configuration is deprecated! Automatically converting to OmegaConf"
            )
            cfg = convert_namespace_to_omegaconf(cfg)

        self.cfg = cfg
        self.task = task

        # Set device: use CUDA if available and not forced to CPU.
        self.cuda = torch.cuda.is_available() and not cfg.common.cpu
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # There is no distributed training in this single GPU setting.
        # Hard-code data parallel properties.
        self._data_parallel_world_size = 1
        self._data_parallel_rank = 0

        # Copy model and criterion to the proper device/dtype.
        self._criterion = criterion
        self._model = model
        if cfg.common.fp16:
            assert not cfg.common.amp, "Cannot use fp16 and AMP together"
            self._criterion = self._criterion.half()
            self._model = self._model.half()
        elif cfg.common.bf16:
            self._criterion = self._criterion.to(dtype=torch.bfloat16)
            self._model = self._model.to(dtype=torch.bfloat16)
        elif cfg.common.amp:
            self._amp_retries = 0

        self._criterion = self._criterion.to(device=self.device)
        self._model = self._model.to(device=self.device)

        # Shared parameter check (if any parameters are shared in the model)
        shared_params = _catalog_shared_params(model)
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info("detected shared parameter: {} <- {}".format(shared_param[0], path))
                _set_module_by_path(self._model, path, ref)

        self._dummy_batch = None
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None
        self._ema = None

        self.quantizer = quantizer
        if self.quantizer is not None:
            self.quantizer.set_trainer(self)

        metrics.log_start_time("wall", priority=790, round=0)
        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    # ------------------------
    # Distributed properties replaced with single GPU settings.
    # ------------------------

    @property
    def data_parallel_world_size(self):
        return self._data_parallel_world_size

    @property
    def data_parallel_rank(self):
        return self._data_parallel_rank

    @property
    def is_data_parallel_master(self):
        return True

    @property
    def use_distributed_wrapper(self) -> bool:
        # No distributed wrapper in single GPU mode.
        return False

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        # Always save checkpoint on single GPU.
        return True

    @property
    def always_call_state_dict_during_save_checkpoint(self) -> bool:
        return False

    @property
    def checkpoint_suffix(self) -> str:
        return self.cfg.checkpoint.checkpoint_suffix or ""

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def ema(self):
        if self._ema is None:
            self._build_ema()
        return self._ema

    def _build_ema(self):
        if self.cfg.ema.store_ema:
            self._ema = build_ema(self._model, self.cfg.ema, self.device)
            logger.info("Exponential Moving Average Shadow Model is initialized.")

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # This initializes lr_scheduler.
        return self._lr_scheduler

    def _build_optimizer(self):
        # Gather all trainable parameters.
        if self.cfg.optimization.debug_param_names and self.cfg.common.fp16_no_flatten_grads:
            params = []
            self.param_names = []
            for n, p in chain(self.model.named_parameters(), self.criterion.named_parameters()):
                if p.requires_grad:
                    params.append(p)
                    self.param_names.append(n)
        else:
            params = list(
                filter(lambda p: p.requires_grad, chain(self.model.parameters(), self.criterion.parameters())))

        # Build optimizer based on precision settings.
        if self.cfg.common.fp16 or self.cfg.common.bf16 or self.cfg.common.amp:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16 or --amp, switching to FP32.")
            if self.cfg.common.memory_efficient_fp16 or self.cfg.common.memory_efficient_bf16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.cfg, params)
            elif self.cfg.common.amp:
                self._optimizer = optim.AMPOptimizer.build_optimizer(self.cfg, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.cfg, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info("NOTE: your device may support faster training with --fp16 or --amp")
            self._optimizer = optim.build_optimizer(self.cfg.optimizer, params)

        # Build the learning rate scheduler.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.cfg.lr_scheduler, self.optimizer)
        self._lr_scheduler.step_update(0)

    @property
    def is_fsdp(self):
        # In single GPU mode, fully sharded DDP is not used.
        return False

    def consolidate_optimizer(self):
        # In single GPU mode, no consolidation is required.
        pass

    def state_dict(self):
        state_dict = {
            "args": None,  # legacy
            "cfg": OmegaConf.to_container(self.cfg, resolve=True, enum_to_str=True)
            if OmegaConf.is_config(self.cfg)
            else self.cfg,
            "model": self.model.state_dict(),
            "criterion": self.criterion.state_dict() if utils.has_parameters(self.criterion) else None,
            "optimizer_history": (self._optim_history or [])
                                 + [
                                     {
                                         "criterion_name": self.get_criterion().__class__.__name__,
                                         "optimizer_name": self.optimizer.__class__.__name__,
                                         "lr_scheduler_state": self.lr_scheduler.state_dict(),
                                         "num_updates": self.get_num_updates(),
                                     }
                                 ],
            "task_state": self.task.state_dict() if self.task is not None else {},
            "extra_state": {
                "metrics": metrics.state_dict(),
                "previous_training_time": self.cumulative_training_time(),
            },
        }
        if self.cfg.ema.store_ema:
            state_dict["extra_state"]["ema"] = self.ema.get_model().state_dict()
            if self.cfg.ema.ema_fp32:
                state_dict["extra_state"]["ema_fp32_params"] = self.ema.fp32_params
        if not self.cfg.checkpoint.no_save_optimizer_state:
            state_dict["last_optimizer_state"] = self.optimizer.state_dict()
        return state_dict

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if self.should_save_checkpoint_on_current_rank:
            logger.info(f"Saving checkpoint to {os.path.abspath(filename)}")
            state_dict = utils.move_to_cpu(self.state_dict())
            state_dict["extra_state"].update(extra_state)
            checkpoint_utils.torch_persistent_save(
                state_dict,
                filename,
                async_write=self.cfg.checkpoint.write_checkpoints_asynchronously,
            )
            logger.info(f"Finished saving checkpoint to {os.path.abspath(filename)}")
            return os.path.abspath(filename)
        return None

    def load_checkpoint(self, filename, reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None,
                        reset_meters=False):
        """
        Load all training state from a checkpoint file.
        In single GPU mode, the checkpoint is loaded directly.
        """
        extra_state, self._optim_history, last_optim_state = None, [], None
        logger.info(f"Preparing to load checkpoint {filename}")
        if PathManager.isfile(filename):
            state = checkpoint_utils.load_checkpoint_to_cpu(filename, load_on_all_dp_ranks=True)
            last_optim_state = state.get("last_optimizer_state", None)

            if state is None:
                logger.info("No checkpoint found at {}".format(filename))
            else:
                try:
                    if "optimizer_history" in state and len(state["optimizer_history"]) > 0 and "num_updates" in \
                            state["optimizer_history"][-1]:
                        self.model.set_num_updates(state["optimizer_history"][-1]["num_updates"])
                    self.model.load_state_dict(state["model"], strict=True, model_cfg=self.cfg.model)
                    del state["model"]
                    if utils.has_parameters(self.get_criterion()):
                        self.get_criterion().load_state_dict(state["criterion"], strict=True)
                        del state["criterion"]
                except Exception:
                    raise Exception(
                        "Cannot load model parameters from checkpoint {}; please ensure that the architectures match.".format(
                            filename))
                extra_state = state["extra_state"]
                self._optim_history = state["optimizer_history"]
        else:
            logger.info("No existing checkpoint found {}".format(filename))
        if last_optim_state is not None and not reset_optimizer:
            self._build_optimizer()
            last_optim = self._optim_history[-1]
            assert last_optim["criterion_name"] == self.get_criterion().__class__.__name__, \
                f"Criterion does not match; please reset the optimizer ({last_optim['criterion_name']} vs {self.get_criterion().__class__.__name__})"
            assert last_optim["optimizer_name"] == self.optimizer.__class__.__name__, \
                f"Optimizer does not match; please reset the optimizer ({last_optim['optimizer_name']} vs {self.optimizer.__class__.__name__})"
            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)
            self.set_num_updates(last_optim["num_updates"])
        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]
            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()
            self.lr_step(epoch)
            if itr_state.get("version", 1) >= 2 and itr_state["iterations_in_epoch"] == 0:
                reset_meters = True
            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()
            if self.cfg.ema.store_ema:
                if "ema" not in extra_state:
                    logger.warn(
                        "EMA not found in checkpoint. But store_ema is True. EMA is re-initialized from checkpoint.")
                    self.ema.restore(state["model"], build_fp32_params=self.cfg.ema.ema_fp32)
                else:
                    logger.info("Loading EMA from checkpoint")
                    self.ema.restore(extra_state["ema"], build_fp32_params=False)
                    if self.cfg.ema.ema_fp32:
                        if "ema_fp32_params" in extra_state:
                            logger.info("Loading EMA fp32 params from checkpoint")
                            self.ema.build_fp32_params(extra_state["ema_fp32_params"])
                        else:
                            logger.info("Building EMA fp32 params from EMA model in checkpoint")
                            self.ema.build_fp32_params()
            logger.info("Loaded checkpoint {} (epoch {} @ {} updates)".format(filename, epoch, self.get_num_updates()))
        return extra_state

    def get_train_iterator(self, epoch, combine=True, load_dataset=True, data_selector=None, shard_batch_itr=True,
                           disable_iterator_cache=False):
        """Return an iterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("Loading train data for epoch {}".format(epoch))
            self.task.load_dataset(self.cfg.dataset.train_subset, epoch=epoch, combine=combine,
                                   data_selector=data_selector, tpu=False)
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.train_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(self.task.max_positions(), self.model.max_positions(),
                                                      self.cfg.dataset.max_tokens),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=1,
            shard_id=0,
            num_workers=self.cfg.dataset.num_workers,
            epoch=epoch,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=self.cfg.optimization.skip_remainder_batch,
            grouped_shuffling=self.cfg.dataset.grouped_shuffling,
            update_epoch_batch_itr=self.cfg.dataset.update_epoch_batch_itr,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def get_valid_iterator(self, subset, disable_iterator_cache=False):
        """Return an iterator over the validation set for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.cfg.dataset.max_tokens_valid,
            max_sentences=self.cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(self.task.max_positions(), self.model.max_positions()),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=1,
            shard_id=0,
            num_workers=self.cfg.dataset.num_workers,
            epoch=1,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=False,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("Begin training epoch {}".format(epoch))
        self.lr_step_begin_epoch(epoch)
        if self.quantizer is not None:
            self.quantizer.begin_epoch(epoch)
        self.task.begin_epoch(epoch, self.get_model())

    def begin_valid_epoch(self, epoch):
        """Called at the beginning of each validation epoch."""
        self.task.begin_valid_epoch(epoch, self.get_model())

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Perform a forward, backward, and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()
        metrics.log_start_time("train_wall", priority=800, round=0)

        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        has_oom = False
        logging_outputs, sample_size, ooms = [], 0, 0

        for i, sample in enumerate(samples):
            sample, is_dummy_batch = self._prepare_sample(sample)
            try:
                # No need for a no_sync context in single GPU mode.
                loss, sample_size_i, logging_output = self.task.train_step(
                    sample=sample,
                    model=self.model,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    update_num=self.get_num_updates(),
                    ignore_grad=is_dummy_batch,
                    **extra_kwargs,
                )
                del loss
                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM encountered: {}".format(e))
                    has_oom = True
                    if raise_oom:
                        raise e
                else:
                    raise e

            if has_oom:
                logger.warning("Recovering from OOM in forward/backward pass")
                ooms += 1
                self.zero_grad()
                if self.cuda:
                    torch.cuda.empty_cache()
                return None

        if is_dummy_batch:
            sample_size = 0.0 if not torch.is_tensor(sample_size) else sample_size.zero_()

        sample_size = float(sample_size) if not torch.is_tensor(sample_size) else sample_size.float()

        # In single GPU mode, simply proceed to update without all-reducing gradients.
        try:
            self.optimizer.multiply_grads(1.0 / (sample_size or 1.0))
            grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm)
            self.task.optimizer_step(self.optimizer, model=self.model, update_num=self.get_num_updates())
            if self.cfg.common.amp and hasattr(self, "_amp_retries") and self._amp_retries:
                self._amp_retries = 0
        except Exception as e:
            logger.error("Error during optimizer step: {}".format(e))
            raise e

        self.set_num_updates(self.get_num_updates() + 1)
        if self.cfg.ema.store_ema:
            self.ema.step(self.get_model(), self.get_num_updates())
            metrics.log_scalar("ema_decay", self.ema.get_decay(), priority=10000, round=5, weight=0)
        metrics.log_stop_time("train_wall")
        return self._reduce_and_log_stats(logging_outputs, sample_size, grad_norm)

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Perform a forward pass in evaluation mode."""
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()
            sample, is_dummy_batch = self._prepare_sample(sample)
            try:
                _loss, sample_size, logging_output = self.task.valid_step(sample, self.model, self.criterion,
                                                                          **extra_kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM during validation: {}".format(e))
                    if not raise_oom:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e
            logging_outputs = [logging_output]
            if is_dummy_batch:
                sample_size = 0.0 if not torch.is_tensor(sample_size) else sample_size.zero_()
        return self._reduce_and_log_stats(logging_outputs, sample_size)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        return self.lr_step_update()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
            new_lr = next(iter(new_lr.values()))
        else:
            metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Return the underlying model instance."""
        return self._model

    def get_criterion(self):
        """Return the underlying criterion instance."""
        return self._criterion

    def get_num_updates(self):
        """Return the number of parameter updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameter updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        if self.quantizer:
            self.quantizer.step_update(self._num_updates)
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(self, clip_norm):
        return self.optimizer.clip_grad_norm(clip_norm)

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            return time.time() - self._start_time + self._previous_training_time
        else:
            return self._cumulative_training_time

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            return t.to(dtype=torch.half) if t.dtype is torch.float32 else t

        def apply_bfloat16(t):
            return t.to(dtype=torch.bfloat16) if t.dtype is torch.float32 else t

        if self.cfg.common.fp16:
            sample = utils.apply_to_sample(apply_half, sample)
        if self.cfg.common.bf16:
            sample = utils.apply_to_sample(apply_bfloat16, sample)
        return sample

    def _prepare_sample(self, sample, is_dummy=False):
        if sample == "DUMMY":
            raise Exception("Uninitialized dummy batch encountered.")
        if sample is None or len(sample) == 0:
            assert self._dummy_batch is not None and len(self._dummy_batch) > 0, "Invalid dummy batch: {}".format(
                self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True
        if self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)
        if self.cuda:
            sample = utils.move_to_cuda(sample)
        if not self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)
        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample
        return sample, False

    def _set_seed(self):
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        if grad_norm is not None:
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.cfg.optimization.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(grad_norm > self.cfg.optimization.clip_norm, grad_norm.new_tensor(100),
                                grad_norm.new_tensor(0)),
                    priority=500,
                    round=1,
                )
        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_criterion())
                del logging_outputs
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value; this may break some functionality")
                metrics.log_scalar("loss", -1)
            logging_output = agg.get_smoothed_values() if not self.cuda else {}
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]
            return logging_output

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()


def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    for name in path.split("."):
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path_parts = path.split(".")
    for name in path_parts[:-1]:
        module = getattr(module, name)
    setattr(module, path_parts[-1], value)
