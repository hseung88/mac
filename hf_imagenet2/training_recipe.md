# Training recipes 

We provide the specific commands and hyperparameters for ResNets and ConvNexts in this recipe.

## ResNet-18
This is the Pytorch default setting to train ResNet-18.

```python
python main.py --arch resnet18 --optimizer adaact \
	--lr 0.1 --beta1 0.9 --beta2 0.999 --eps 1e-8 --wd 0.002 \
	--batch-size 256 --workers 8 --epochs 90
```


## ResNet-50
This is a default setting used to train [ResNets](https://arxiv.org/abs/2110.00476).

```python
python ./train.py --model resnet50 --sched cosine --epochs 200 \
    --opt adaact --lr 0.1 --opt-betas 0.9 0.999 --opt-eps 1e-8 --weight-decay 0.002 \
    --workers 16 --warmup-epochs 5 --warmup-lr 1e-2 --min-lr 1e-5 --batch-size 256 \
    --grad-accum-steps 8 --amp --aug-repeats 0  \
    --aa rand-m7-mstd0.5-inc1 --smoothing 0.0 --remode pixel --crop-pct 0.95 \
    --reprob 0.0 --drop 0.0 --drop-path 0.05 --mixup 0.1 --cutmix 1.0
```


## ResNet-101

```python
python ./train.py --model resnet101 --sched cosine --epochs 200 \
    --opt adaact --lr 0.1 --opt-betas 0.9 0.999 --opt-eps 1e-8 --weight-decay 0.002 \
    --workers 16 --warmup-epochs 5 --warmup-lr 1e-2 --min-lr 1e-5 --batch-size 256 \
    --grad-accum-steps 8 --amp --aug-repeats 0  \
    --aa rand-m7-mstd0.5-inc1 --smoothing 0.0 --remode pixel --crop-pct 0.95 \
    --reprob 0.0 --drop 0.0 --drop-path 0.05 --mixup 0.1 --cutmix 1.0
```


## ConvNext-Tiny

This is a default setting to train ConvNext.

```python
torchrun --nproc_per_node=4 ./train.py --model convnext_tiny_hnf --sched cosine --epochs 150 \
    --opt adaact --lr 0.1 --opt-betas 0.9 0.999 --opt-eps 1e-8 --weight-decay 0.002 \
    --workers 8 --warmup-epochs 5 --warmup-lr 1e-2 --min-lr 1e-5 --batch-size 256 \
    --grad-accum-steps 2 --amp --aug-repeats 0 --aa rand-m7-mstd0.5-inc1 \
    --smoothing 0.1 --remode pixel --reprob 0.25 --drop 0.0 --drop-path 0.1 \
    --mixup 0.8 --cutmix 1.0 --model-ema --train-interpolation random
```

## ConvNext-Small

```python
torchrun --nproc_per_node=4 ./train.py --model convnext_small --sched cosine --epochs 150 \
    --opt adaact --lr 0.1 --opt-betas 0.9 0.999 --opt-eps 1e-8 --weight-decay 0.002 \
    --workers 8 --warmup-epochs 5 --warmup-lr 1e-2 --min-lr 1e-5 --batch-size 256 \
    --grad-accum-steps 2 --amp --aug-repeats 0 --aa rand-m7-mstd0.5-inc1 \
    --smoothing 0.1 --remode pixel --reprob 0.25 --drop 0.0 --drop-path 0.1 \
    --mixup 0.8 --cutmix 1.0 --model-ema --train-interpolation random
```