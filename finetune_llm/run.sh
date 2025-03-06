#!/bin/bash

# usage : ./scripts/hparam_search.sh 0 distilbert ZOAdam sst2 1e-03
CUDA_VISIBLE_DEVICES_ARG=$1
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_ARG
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

model=$2
algorithm=$3
task=$4
adaptivity=$5

if [ $model == "distilbert" ]; then
    model=distilbert-base-cased
    if [ $algorithm == "ZO" ]; then
        lr=1e-6
        epochs=25000
    elif [ $algorithm == "ZOSVRG" ]; then
        lr=1e-3
        epochs=6250
    elif [ $algorithm == "ZOAdam" ]; then
        lr=1e-4
        epochs=25000
    fi

elif [ $model == "roberta-large" ]; then
    if [ $algorithm == "ZO" ]; then
        lr=1e-6
        epochs=12000
    elif [ $algorithm == "ZOSVRG" ]; then
        lr=5e-5
        epochs=3000
    elif [ $algorithm == "ZOAdam" ]; then
        lr=5e-5
        epochs=12000
    fi

elif [ $model == "gpt2-xl" ]; then
    model=openai-community/gpt2-xl
    if [ $algorithm == "ZO" ]; then
        lr=5e-6
        epochs=4000
    elif [ $algorithm == "ZOSVRG" ]; then
        lr=5e-5
        epochs=1000
    elif [ $algorithm == "ZOAdam" ]; then
        lr=2e-4
        epochs=1000
    fi

elif [ $model == "opt-2.7b" ]; then
    model=facebook/opt-2.7b
    if [ $algorithm == "ZO" ]; then
        lr=5e-6
        epochs=4000
    elif [ $algorithm == "ZOSVRG" ]; then
        lr=5e-5
        epochs=1000
    elif [ $algorithm == "ZOAdam" ]; then
        lr=5e-5
        epochs=1000
    fi
fi

bsz=64
bsz_lim=64


echo "Running experiment with $model, $task, $algorithm, lr=$lr, adaptivity=$adaptivity"

python run.py\
    --epochs $epochs\
    --model_name $model\
    --task $task\
    --algorithm $algorithm\
    --lr $lr --lr_mezosvrg_mb 1e-6 --adaptivity $adaptivity\
    --perturbation_scale 1e-3 --q 2 --anneal 5\
    --samplesize 512 --samplesize_validation 256\
    --batchsize $bsz --batchsize_limit $bsz_lim\
    --full_parameter\
    --max_seq_length 128\
    --device 0 --save_every 10000000 --logging wandb\
    --init_seed 42 --trial 0 --early_stopping --patience 1000\
    --low_bit_adam 8