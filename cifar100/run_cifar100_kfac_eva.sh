#!/bin/bash

# List of commands
commands=(
"python main.py --model resnet110 --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model resnet110 --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 100 --run"
"python main.py --model resnet110 --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 200 --run"
"python main.py --model resnet110 --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model resnet110 --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 100 --run"
"python main.py --model resnet110 --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 200 --run"
"python main.py --model densenet --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model densenet --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 100 --run"
"python main.py --model densenet --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 200 --run"
"python main.py --model densenet --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model densenet --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 100 --run"
"python main.py --model densenet --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 200 --run"
"python main.py --model wrn --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model wrn --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 100 --run"
"python main.py --model wrn --optim kfac --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 200 --run"
"python main.py --model wrn --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 50 --run"
"python main.py --model wrn --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 100 --run"
"python main.py --model wrn --optim eva --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --stat_decay 0.95 --damping 0.03 --tcov 5 --tinv 50 --epoch 200 --run"
)

# Loop through commands
for run in {0..4}; do
    for cmd in "${commands[@]}"; do
        eval "$cmd $run;"
    done
done