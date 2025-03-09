#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Pretraining a Seq2Seq model from scratch with custom optimizer choices.
This script is adapted from the translation example to support from-scratch pretraining and comparing optimizers.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import datasets
import evaluate

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torch
from optimizers.mac import MAC

# Check minimum version requirements
#check_min_version("4.50.0.dev0")
#require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)


# ----------------------
# Argument definitions
# ----------------------

@dataclass
class ModelArguments:
    """
    Arguments for model configuration.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Model identifier or path. For from-scratch pretraining, provide a model id that corresponds to a config."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the downloaded models."}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use one of the fast tokenizers."}
    )
    model_revision: str = field(
        default="main", metadata={"help": "The specific model version to use."}
    )
    token: Optional[str] = field(
        default=None, metadata={"help": "Token for remote files, if necessary."}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Whether to trust remote code execution from the Hub."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to use for pretraining.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (e.g., a plain text file with one sentence per line)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional evaluation data file."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of processes for preprocessing."}
    )
    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "Maximum input sequence length after tokenization."}
    )
    pad_to_max_length: bool = field(
        default=False, metadata={"help": "Whether to pad all samples to the model maximum length."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of training examples for debugging."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of evaluation examples for debugging."}
    )
    # Additional fields for translation tasks
    source_lang: Optional[str] = field(
        default=None, metadata={"help": "Source language id for translation."}
    )
    target_lang: Optional[str] = field(
        default=None, metadata={"help": "Target language id for translation."}
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")


@dataclass
class OptimizerArguments:
    """
    Arguments for choosing the optimizer.
    """
    optimizer_type: str = field(
        default="adamw", metadata={"help": "optimizer to use"}
    )
    lr: float = field(
        default=0.001, metadata={"help": "learning rate"}
    )
    momentum: float = field(
        default=0.9, metadata={"help": "momentum"}
    )
    stat_decay: float = field(
        default=0.95, metadata={"help": "stat decay"}
    )
    damping: float = field(
        default=1.0, metadata={"help": "damping"}
    )
    tcov: int = field(
        default=5, metadata={"help": "tcov"}
    )
    tinv: float = field(
        default=5.0, metadata={"help": "tinv"}
    )



# ----------------------
# Custom Trainer
# ----------------------

class CustomTrainer(Seq2SeqTrainer):
    """
    Custom trainer that allows switching among different optimizers.
    """

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_type = getattr(self.args, "optimizer_type", "adamw").lower()
            if optimizer_type == "sgd":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                              momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            elif optimizer_type == "adamw":
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
            elif optimizer_type == "mac":
                optimizer = MAC(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                stat_decay=self.args.stat_decay, weight_decay=self.args.weight_decay,
                                damping=self.args.damping, Tcov=self.args.tcov, Tinv=self.args.tinv)
            else:
                logger.warning(f"Unknown optimizer type '{optimizer_type}'. Defaulting to AdamW.")
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            self.optimizer = optimizer
        return self.optimizer


# ----------------------
# Main function
# ----------------------

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, OptimizerArguments))

    # Support parsing from a JSON file as well.
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, optimizer_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, optimizer_args = parser.parse_args_into_dataclasses()

    # Merge optimizer arguments into training args so we can access them in the trainer.
    setattr(training_args, "optimizer_type", optimizer_args.optimizer_type)

    # Sending telemetry data (optional)
    send_example_telemetry("pretraining_custom_optimizers", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())

    # Detect last checkpoint if any exists.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) exists and is not empty. Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # ---------------------------
    # Model & Tokenizer Setup
    # ---------------------------
    # Create config and initialize model from scratch (random weights)
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # For pretraining from scratch, we initialize without loading pretrained weights.
    model = AutoModelForSeq2SeqLM.from_config(config)

    # Load tokenizer (this can be a pretrained tokenizer even if the model is random)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # ---------------------------
    # Dataset Preparation
    # ---------------------------
    if data_args.dataset_name is not None:
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        raw_datasets = datasets.load_dataset("text", data_files=data_files, cache_dir=model_args.cache_dir)

    # We assume the dataset has a "text" field. Adjust if needed.
    column_names = raw_datasets["train"].column_names if training_args.do_train else raw_datasets[
        "validation"].column_names
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = examples["text"]
        model_inputs = tokenizer(texts, max_length=data_args.max_source_length, truncation=True, padding=padding)
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    else:
        eval_dataset = None

    # ---------------------------
    # Data Collator
    # ---------------------------
    data_collator = default_data_collator if data_args.pad_to_max_length else DataCollatorForSeq2Seq(
        tokenizer, model=model, pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # ---------------------------
    # Metric (optional)
    # ---------------------------
    # For pretraining, you might not have a metric.
    metric = evaluate.load("bleu") if training_args.do_eval else None

    def compute_metrics(eval_preds):
        # Dummy compute metrics; adjust as needed for your pretraining objective.
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # For demonstration, we calculate BLEU score if metric is available.
        result = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        result = {"bleu": result["score"]}
        return result

    # ---------------------------
    # Initialize Trainer
    # ---------------------------
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
    )

    # ---------------------------
    # Training
    # ---------------------------
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves model and tokenizer

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset) if train_dataset is not None else 0
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ---------------------------
    # Evaluation
    # ---------------------------
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset) if eval_dataset is not None else 0
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        results = metrics

    # Optionally, you could add a prediction phase if needed.
    return results


def _mp_fn(index):
    # For TPUs: entry point for subprocesses.
    main()


if __name__ == "__main__":
    main()
