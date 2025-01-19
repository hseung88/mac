#!/bin/bash

# List of commands
commands=(
"python main.py --model resnet110 --optim mac --lr 0.01 --momentum 0.9 --weight_decay 0.05 --damping 0.1 --tinv 100 --epoch 50 --run"
"python main.py --model resnet110 --optim mac --lr 0.01 --momentum 0.9 --weight_decay 0.05 --damping 0.1 --tinv 100 --epoch 100 --run"
"python main.py --model resnet110 --optim mac --lr 0.01 --momentum 0.9 --weight_decay 0.05 --damping 0.1 --tinv 100 --epoch 200 --run"
"python main.py --model densenet --optim mac --lr 0.01 --momentum 0.9 --weight_decay 0.05 --damping 0.1 --tinv 100 --epoch 50 --run"
"python main.py --model densenet --optim mac --lr 0.01 --momentum 0.9 --weight_decay 0.05 --damping 0.1 --tinv 100 --epoch 100 --run"
"python main.py --model densenet --optim mac --lr 0.01 --momentum 0.9 --weight_decay 0.05 --damping 0.1 --tinv 100 --epoch 200 --run"
)

# Loop through commands
for run in {0..4}; do
    for cmd in "${commands[@]}"; do
        eval "$cmd $run;"
    done
done