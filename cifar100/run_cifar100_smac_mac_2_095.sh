#!/bin/bash

# List of commands
commands=(
"python main.py --model resnet110 --optim smac --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 200 --batchsize 128 --run"
"python main.py --model resnet110 --optim mac --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 200 --batchsize 128 --run"
"python main.py --model densenet --optim smac --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 200 --batchsize 128 --run"
"python main.py --model densenet --optim mac --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 200 --batchsize 128 --run"
"python main.py --model wrn --optim smac --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 200 --batchsize 128 --run"
"python main.py --model wrn --optim mac --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 200 --batchsize 128 --run"
)

# Loop through commands
for run in {0..4}; do
    for cmd in "${commands[@]}"; do
        eval "$cmd $run;"
    done
done