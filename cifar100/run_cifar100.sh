#!/bin/bash

# List of commands
commands=(
"python main.py --model resnet110 --optim mac --lr 0.1 --momentum 0.9 --stat_decay 0.999 --damping 1.0 --weight_decay 0.0005 --update 50 --epoch 50 --run"
"python main.py --model resnet110 --optim smac --lr 0.1 --momentum 0.9 --stat_decay 0.999 --damping 1.0 --weight_decay 0.0005 --update 50 --epoch 50 --run"
"python main.py --model resnet110 --optim sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --epoch 50 --run"
"python main.py --model resnet110 --optim adam --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0.0005 --epoch 50 --run"
"python main.py --model resnet110 --optim adamw --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0.5 --epoch 50 --run"
"python main.py --model resnet110 --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model resnet110 --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model resnet110 --optim foof --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model resnet110 --optim nysact --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --rank 10 --epoch 50 --run"
"python main.py --model resnet110 --optim shaper --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 50 --run"
)

# Loop through commands
for run in {0..4}; do
    for cmd in "${commands[@]}"; do
        eval "$cmd $run;"
    done
done