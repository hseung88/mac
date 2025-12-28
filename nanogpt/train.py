"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import pickle
import time
import csv
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText

# I/O
out_dir = "out"
eval_interval = 1000
eval_iters = 200
log_interval = 10
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2_50M"  # 'run' + str(time.time())

# data
dataset = "openwebtext"
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512

# model
n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?

# optimization hyperparameters
optimizer = "mac"  # 'adamw' or 'mac'
learning_rate = 1e-3  # max learning rate
max_iters = 15000  # total number of training iterations
weight_decay = 1e-1

# adamw optimizer
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# MAC optimizer
momentum = 0.9
stat_decay = 0.95
damping = 1.0
Tcov = 5
Tinv = 5

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 15000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# ---------------- CSV logging (master only) ----------------
csv_log_path = os.path.join(out_dir, "metrics.csv")
if master_process and (not os.path.exists(csv_log_path)):
    with open(csv_log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iter",
                "tokens_total",
                "train_loss",
                "val_loss",
                "lr",
                "iter_time_sec",
                "iter_time_ema_sec",
            ]
        )


def append_csv_row(
    iter_i: int,
    tokens_total: int,
    train_loss: float,
    val_loss: float,
    lr_val: float,
    iter_time_sec: float,
    iter_time_ema_sec: float,
):
    if not master_process:
        return
    with open(csv_log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                int(iter_i),
                int(tokens_total),
                float(train_loss),
                float(val_loss),
                float(lr_val),
                float(iter_time_sec),
                float(iter_time_ema_sec),
            ]
        )
        f.flush()
        os.fsync(f.fileno())


# -----------------------------------------------------------

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def train_loader(count_batch_size=102400, max_num_batches=2):
    data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    batch_block_len = count_batch_size * block_size

    batch_processed = 0
    for i in range(0, len(data), batch_block_len):
        flat_batch = data[i : i + batch_block_len].astype(np.int64)
        yield flat_batch
        batch_processed += 1
        if batch_processed >= max_num_batches:
            break


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Unknown init_from: {init_from}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

opt_args = SimpleNamespace(
    optimizer=optimizer.lower(),
    learning_rate=learning_rate,
    max_iters=max_iters,
    weight_decay=weight_decay,
    beta1=beta1,
    beta2=beta2,
    grad_clip=grad_clip,
    momentum=momentum,
    stat_decay=stat_decay,
    damping=damping,
    Tcov=Tcov,
    Tinv=Tinv,
    device_type=device_type,
)

# optimizer
optimizer = model.configure_optimizers(opt_args)

if opt_args.optimizer in ("mac", "smac"):
    # optimizer._configure(train_loader, model, device)
    optimizer.model = model

if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

# timing EMA for per-iteration wall time
dt_ema = float("nan")
ema_beta = 0.98
last_dt = float("nan")

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        tr_loss = float(losses["train"])
        va_loss = float(losses["val"])
        tokens_total = int(iter_num * tokens_per_iter)

        print(f"step {iter_num}: train loss {tr_loss:.4f}, val loss {va_loss:.4f}")

        # CSV log (eval points)
        append_csv_row(
            iter_i=iter_num,
            tokens_total=tokens_total,
            train_loss=tr_loss,
            val_loss=va_loss,
            lr_val=float(lr),
            iter_time_sec=float(last_dt),
            iter_time_ema_sec=float(dt_ema),
        )

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "tokens_total": tokens_total,
                    "train/loss": tr_loss,
                    "val/loss": va_loss,
                    "lr": lr,
                    "iter_time_sec": last_dt,
                    "iter_time_ema_sec": dt_ema,
                    "mfu": running_mfu * 100,
                }
            )

        if va_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = va_loss
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    last_dt = t1 - t0
    t0 = t1
    if math.isnan(dt_ema):
        dt_ema = last_dt
    else:
        dt_ema = ema_beta * dt_ema + (1.0 - ema_beta) * last_dt

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, last_dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {last_dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}%"
        )

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num >= max_iters:
        break

if ddp:
    destroy_process_group()
