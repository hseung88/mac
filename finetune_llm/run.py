import os, sys
import time
import math
import torch
import pickle
import logging
import argparse

import numpy as np
import transformers
import pytorch_lightning as pl

from tqdm import tqdm
from termcolor import colored

from utils import GLUEDataModule, SuperGLUEDataModule, GLUE_TASKS, SUPERGLUE_TASKS
from llm_module import GLUETransformer

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Model Fine-tuning')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--samplesize', type=int, default=1024, help='Training data sample size')
    parser.add_argument('--samplesize_validation', type=int, default=128, help='Validation data sample size')
    parser.add_argument('--model_name', type=str, default='DistilBert', help='Name of the pre-trained model')
    parser.add_argument('--task', type=str, default='mnli', help='Task for model training')
    parser.add_argument('--full_parameter', action='store_true', help='True for full parameter fine-tuning')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batchsize_limit', type=int, default=64,
                        help='Max batch size to be used to avoid memory error')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Max sequence length for inputs')
    parser.add_argument('--anneal', type=float, default=1.5, help='Annealing parameter')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=int, default=0, help='GPU Number')
    parser.add_argument('--results', type=str, default='results_demo', help='Name of folder to store results')
    parser.add_argument('--soft_prompt', action='store_true', help='True for using soft prompt')
    parser.add_argument('--logging', type=str, default="wandb",
                        help="Choose logging method; either wandb or tensorboard or none")
    parser.add_argument('--low_bit_adam', type=int, default=0, help='Use Adam with quantized states; options: 4 or 8')
    parser.add_argument('--trial', type=int, default=0, help='Trial number')
    parser.add_argument('--init_seed', type=int, default=None, help='Random seed for model initialization')

    # Arguments for MAC optimizer
    parser.add_argument('--optimizer', type=str, default='mac', help='Optimizer to use: mac, sgd, or adam')
    parser.add_argument('--stat_decay', type=float, default=0.95, help='Statistic decay for MAC optimizer')
    parser.add_argument('--tcov', type=int, default=5, help='Tcov parameter for MAC optimizer')
    parser.add_argument('--tinv', type=int, default=5, help='Tinv parameter for MAC optimizer')
    parser.add_argument('--damping', type=float, default=1.0, help='Damping for MAC optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')

    args = parser.parse_args()
    return args


def save_pickle(data, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)


def get_data_module(args):
    if args.task in GLUE_TASKS:
        dm_class = GLUEDataModule
    elif args.task in SUPERGLUE_TASKS:
        dm_class = SuperGLUEDataModule
    else:
        raise ValueError(f"Task {args.task} is not supported")
    dm = dm_class(
        model_name_or_path=args.model_name,
        task_name=args.task,
        max_seq_length=args.max_seq_length,
        sample_size=args.samplesize,
        train_batch_size=args.batchsize,
        validation_sample_size=args.samplesize_validation,
        eval_batch_size=args.batchsize,
        soft_prompt=args.soft_prompt,
        hf_token=os.getenv('HF_TOKEN')
    )
    dm.setup(stage='fit')
    return dm


def get_model(args, dm):
    transformer = GLUETransformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.lr,
        anneal=args.anneal,
        full_parameter=args.full_parameter,
        batchsize_limit=args.batchsize_limit,
        soft_prompt=args.soft_prompt,
        optimizer_name=args.optimizer,
        logging=args.logging,
        hf_token=os.getenv('HF_TOKEN'),
        model_init_seed=args.init_seed,
    )
    return transformer


if __name__ == "__main__":
    args = parse_arguments()
    trimmed_model_name = args.model_name.split('/')[-1]
    num_steps = math.ceil(args.samplesize / args.batchsize) * args.epochs
    args.run_name = f'{trimmed_model_name}_{args.task}_mac_lr{args.lr:.0e}_bsz{args.batchsize}_steps{num_steps}'
    if args.init_seed is not None:
        args.run_name += f'_init_seed_{args.init_seed}'
    args.run_name += f'_trial_{args.trial}'
    args.output_dir = os.path.join('results', trimmed_model_name, args.task, args.run_name)

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = colored('[%(asctime)s %(name)s]', 'green') + \
          colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, f'log_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training Arguments: {args}")

    # Initialize data module and model
    dm = get_data_module(args)
    model = get_model(args, dm)

    # Initialize PyTorch Lightning Trainer for standard training
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=[args.device] if torch.cuda.is_available() else None,
        default_root_dir=args.output_dir,
        logger=None  # Optionally, add a WandbLogger or TensorBoardLogger here.
    )

    trainer.fit(model, datamodule=dm)
