#!/bin/bash

# List of commands
commands=(
"python main.py --model resnet110 --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 50 --batchsize 128 --run"
"python main.py --model resnet110 --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 100 --batchsize 128 --run"
"python main.py --model resnet110 --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 200 --batchsize 128 --run"
"python main.py --model densenet --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 50 --batchsize 128 --run"
"python main.py --model densenet --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 100 --batchsize 128 --run"
"python main.py --model densenet --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 200 --batchsize 128 --run"
"python main.py --model wrn --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 50 --batchsize 128 --run"
"python main.py --model wrn --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 100 --batchsize 128 --run"
"python main.py --model wrn --optim soap --lr 0.001 --beta1 0.9 --beta2 0.999 --weight_decay 0.5 --eps 1e-8 --update 50 --epoch 200 --batchsize 128 --run"
)

# Loop through commands
for run in {0..4}; do
    for cmd in "${commands[@]}"; do
        eval "$cmd $run;"
    done
done