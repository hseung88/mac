import argparse
import math
import os
import random
import numpy as np
from datasets import load_dataset
import evaluate

import torch
from torch.optim import AdamW, SGD
from transformers import (
    EncoderDecoderModel,
    EncoderDecoderConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from functools import partial
from optimizers.mac import MAC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_function(examples, tokenizer, max_source_length=128, max_target_length=128):
    # For IWSLT 2014, 'de' is the German source and 'en' is the English target.
    inputs = examples['de']
    targets = examples['en']
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Custom Trainer: override optimizer and scheduler creation to match Fairseq settings.
class CustomTrainer(Trainer):
    def __init__(self, optimizer_name="adamw", lr_scheduler_type="inverse_sqrt", warmup_steps=4000, **kwargs):
        self.optimizer_name = optimizer_name.lower()
        self.lr_scheduler_type = lr_scheduler_type.lower()
        self.custom_warmup_steps = warmup_steps
        super().__init__(**kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(self.model)
            if self.optimizer_name == "sgd":
                self.optimizer = SGD(optimizer_grouped_parameters, lr=self.args.learning_rate, momentum=0.9)
            elif self.optimizer_name == "adamw":
                self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate,
                                       weight_decay=self.args.weight_decay)
            elif self.optimizer_name == "mac":
                self.optimizer = MAC(optimizer_grouped_parameters, lr=self.args.learning_rate,
                                     momentum=self.args.momentum, stat_decay=self.args.stat_decay,
                                     weight_decay=self.args.weight_decay, damping=self.args.damping,
                                     Tcov=self.args.tcov, Tinv=self.args.tinv)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int):
        # Custom inverse square root scheduler
        if self.lr_scheduler_type == "inverse_sqrt":
            def lr_lambda(current_step):
                if current_step < self.custom_warmup_steps:
                    return float(current_step) / float(max(1, self.custom_warmup_steps))
                return (self.custom_warmup_steps ** 0.5) / (current_step ** 0.5)

            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            return scheduler
        else:
            return super().create_scheduler(num_training_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adamw", "sgd", "mac"],
                        help="Type of optimizer")
    parser.add_argument("--lr_scheduler", type=str, default="inverse_sqrt", choices=["inverse_sqrt", "linear"],
                        help="Type of learning rate scheduler")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--stat_decay', default=1e-4, type=float, help='stat decay')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--damping', default=0.01, type=float, help='damping factor for kfac and foof')
    parser.add_argument('--tcov', default=5, type=int, help='preconditioner update period for kfac and foof')
    parser.add_argument('--tinv', default=50, type=int, help='preconditioner inverse period for kfac and foof')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load the dataset (IWSLT 2014 de-en)
    raw_datasets = load_dataset("iwslt2017", "de-en")

    # 2. Initialize the tokenizer (using t5-small tokenizer as an example;
    # you may choose to train your own tokenizer from scratch)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # 3. Build the Transformer from scratch: settings similar to Fairseq's transformer_iwslt_de_en
    # with dropout set to 0.3.
    encoder_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.3,
        "activation_function": "relu",
        "max_position_embeddings": 1024,
    }
    decoder_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "ffn_dim": 2048,
        "dropout": 0.3,
        "activation_function": "relu",
        "max_position_embeddings": 1024,
        "is_decoder": True,
        "add_cross_attention": True,
    }
    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    model = EncoderDecoderModel(config)

    # Share decoder input and output embeddings (tie weights)
    model.decoder.embed_tokens = model.encoder.embed_tokens
    model.config.tie_word_embeddings = True

    # As we are training from scratch, we use the randomly initialized weights.
    model.config.decoder_start_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
    model.config.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 1
    model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model.config.vocab_size = tokenizer.vocab_size

    # 4. Preprocess the dataset
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # 5. Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 6. Evaluation metric (BLEU) using sacreBLEU
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # sacreBLEU expects references as a list of lists
        decoded_labels = [[ref] for ref in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    # 7. Set up TrainingArguments with logging and evaluation per epoch
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_strategy="epoch",  # Log metrics at the end of each epoch
        save_strategy="no",  # Do not save checkpoints (--no-save)
        learning_rate=args.learning_rate,
        weight_decay=0.0001,  # Weight decay as in Fairseq
        gradient_clip_norm=0.0,  # Clip norm 0.0 (no clipping)
        logging_steps=1000,  # Additional logging every 1000 steps (if needed)
        seed=args.seed,
        do_train=True,
        do_eval=True,
        predict_with_generate=True,  # Enable generation for BLEU evaluation
        num_beams=5,  # Beam search with beam size 5
        max_length=128,  # Generation max length (adjustable)
        label_smoothing_factor=0.1,  # Label smoothing factor as in Fairseq
        warmup_steps=4000,  # Warmup steps set to 4000
    )

    # 8. Create the Trainer with custom optimizer and scheduler
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizer_name=args.optimizer,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=4000,
    )

    # 9. Train and evaluate the model
    trainer.train()
    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)


if __name__ == "__main__":
    main()
