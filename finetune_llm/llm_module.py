import os
import time
import math
import pickle
import logging
import numpy as np
import torch
import wandb

from typing import Optional
from pytorch_lightning import LightningModule
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    OPTForSequenceClassification,
)

from mezo_src.models import RobertaModelForPromptFinetuning
from mezo_src.modeling_roberta import RobertaConfig
from optimizers.mac import MAC

MODEL_NAME = "model.pt"
TRAIN_STATE_NAME = "train_state.pkl"

class TrainState:
    def __init__(self):
        self.tr_loss = []
        self.time = []
        self.global_training_steps = 0
        self.learning_rate = 0

class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        full_parameter: bool = True,
        soft_prompt: bool = False,
        logger_type: str = 'wandb',
        hf_token: Optional[str] = None,
        model_init_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name_or_path
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.soft_prompt = soft_prompt

        if model_init_seed is not None:
            original_seed = torch.initial_seed()
            torch.manual_seed(model_init_seed)

        if model_name_or_path in ['distilbert-base-cased', 'roberta-large']:
            if self.soft_prompt and model_name_or_path == 'roberta-large':
                config = RobertaConfig.from_pretrained(
                    'roberta-large',
                    num_labels=num_labels,
                    finetuning_task=self.hparams.task_name if hasattr(self.hparams, "task_name") else None
                )
                self.model = RobertaModelForPromptFinetuning.from_pretrained(
                    "roberta-large",
                    config=config
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        elif 'gpt2' in model_name_or_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
            self.model.config.pad_token_id = self.model.config.eos_token_id
        elif 'opt' in model_name_or_path:
            self.model = OPTForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        else:
            raise NotImplementedError(f"Model {model_name_or_path} not supported yet.")

        if model_init_seed is not None:
            torch.manual_seed(original_seed)

        logging.info(self.model)

        self.full_parameter = full_parameter
        self.logger_type = logger_type

        self.state = TrainState()
        self.state.learning_rate = self.hparams.learning_rate

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_params(self):
        model = self.model
        if self.full_parameter:
            self.params = [(n, p) for n, p in model.named_parameters()]
        else:
            raise NotImplementedError("Partial optimization not supported yet.")

    def configure_optimizers(self):
        """Return an optimizer based on the chosen optimizer name."""
        model = self.model
        if self.full_parameter:
            self.params_to_opt = model.parameters()
        else:
            raise NotImplementedError("Partial optimization not supported yet.")

        optimizer_name = self.hparams.optimizer_name.lower()
        if optimizer_name == 'mac':
            optimizer = MAC(
                self.params_to_opt,
                lr=self.hparams.learning_rate,
                stat_decay=self.hparams.stat_decay,
                damping=self.hparams.damping,
                Tcov=self.hparams.tcov,
                Tinv=self.hparams.tinv
            )
            optimizer.model = model
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.params_to_opt, lr=self.hparams.learning_rate)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.params_to_opt, lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer specified: {optimizer_name}")

        return optimizer

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.state.tr_loss.append(loss.detach().cpu().float().numpy())
        self.state.global_training_steps += 1
        # Instead of wandb.log, use self.log() or simply omit logging here.
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        correct = (preds == labels).sum().item()
        total = len(labels)
        acc = correct / total
        return {"val_loss": val_loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        outputs = self.trainer.callback_metrics
        avg_loss = outputs.get("val_loss", None)
        avg_acc = outputs.get("val_acc", None)
        logging.info(f"Validation Loss: {avg_loss}, Validation Accuracy: {avg_acc}")

    def load_from_checkpoint(self, checkpoint_path: str):
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, MODEL_NAME)))
        self.state = pickle.load(open(os.path.join(checkpoint_path, TRAIN_STATE_NAME), 'rb'))

    def save_checkpoint(self, checkpoint_path: str):
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_path, MODEL_NAME))
        pickle.dump(self.state, open(os.path.join(checkpoint_path, TRAIN_STATE_NAME), 'wb'))
