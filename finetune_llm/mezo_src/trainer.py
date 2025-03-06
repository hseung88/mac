import os
import math
import time
import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from packaging import version
from torch.optim import SGD
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_utils import TrainOutput
from transformers.utils import logging

from tqdm import tqdm, trange
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer  # your base trainer
from transformers.trainer_callback import DefaultFlowCallback, ProgressCallback, TrainerState

# Optional native AMP support
_use_native_amp = version.parse(torch.__version__) >= version.parse("1.6")
if _use_native_amp:
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

# Import MAC optimizer
from optimizers.mac import MAC

DEFAULT_CALLBACKS = [DefaultFlowCallback, ProgressCallback]

def default_dev_objective(metrics: Dict) -> float:
    if "eval_acc" in metrics:
        return metrics["eval_acc"]
    raise Exception("No metric found for evaluation.")

class Trainer(LinearHeadTrainer):
    """
    A simplified Trainer that uses standard backpropagation.
    The zero-order methods have been removed. Optimizer and scheduler are created
    based on training arguments.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            # Gather parameters (with optional layer freezing)
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except Exception as e:
                            print(n)
                            raise e
                        if layer_num >= self.args.fix_layers:
                            params[n] = p
                    elif 'embeddings' in n:
                        continue
                    else:
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer.lower() == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer.lower() == 'sgd':
                self.optimizer = SGD(optimizer_grouped_parameters, lr=self.args.learning_rate, momentum=0.9)
            elif self.args.optimizer.lower() == 'mac':
                self.optimizer = MAC(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    momentum=0.9,
                    stat_decay=self.args.stat_decay,
                    damping=self.args.damping,
                    Tcov=self.args.tcov,
                    Tinv=self.args.tinv,
                )
                self.optimizer.model = self.model
            else:
                raise NotImplementedError(f"Optimizer {self.args.optimizer} not implemented.")

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point using standard forward/backward updates.
        """
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = max(1, len(train_dataloader) // self.args.gradient_accumulation_steps)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        model = self.model

        if self.args.fp16 and _use_native_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size = %d", self.args.train_batch_size * self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0

        tr_loss = 0.0
        model.zero_grad()
        for epoch in range(num_train_epochs):
            if isinstance(train_dataloader, torch.utils.data.DataLoader) and hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not self.is_local_process_zero())
            for step, inputs in enumerate(epoch_iterator):
                self.model.train()
                inputs = self._prepare_inputs(inputs)
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        loss = self.compute_loss(self.model, inputs)
                else:
                    loss = self.compute_loss(self.model, inputs)
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
            logger.info("Epoch {} completed.".format(epoch + 1))
            self.evaluate()
        logger.info("Training completed.")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, {}), self.state.global_step

    def evaluate(self, eval_dataset=None) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()
        losses = []
        all_preds = []
        all_labels = []
        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(inputs['labels'].cpu().numpy())
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        metrics = {"eval_loss": np.mean(losses), "eval_acc": acc}
        logger.info(metrics)
        return metrics
