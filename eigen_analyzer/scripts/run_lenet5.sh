#!/bin/bash


display_usage() {
    echo "Usage: $0 "
}


# check if -h or --help is supplied
if [[ ( $@ == "--help") || $@ == "-h" ]]
then
    display_usage
    exit 0
fi

 
# get the project root directory
APP_ROOT="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# input argument
dataset=cifar10
network=lenet5

# working dir
cd ${APP_ROOT}

# train and save checkpoints
python main.py network=${network} epochs=10 save=True

# computing eigenvalues
python main.py network=${network} trainer=act_eigendecomp save=True

python main.py network=${network} trainer=preact_eigendecomp save=True
