"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import torch
import torch.nn as nn
import pickle
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from models import *
from torch.optim import Adam, SGD, AdamW
from timm.optim import Adan
from optimizers.kfac import KFAC
from optimizers.foof import FOOF
from optimizers.adaact_v2 import AdaAct
from .eigen_utils import hess_scipy


data_dir = './data'


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epoch', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--model', default='resnet20', type=str, help='model',
                        choices=['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=['sgd', 'adam', 'adamw', 'adan', 'kfac', 'foof', 'adaact'])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for numerical stability')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--stat_decay', default=1e-4, type=float, help='stat decay')
    parser.add_argument('--beta1', default=0.9, type=float, help='moving average coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='moving average coefficients beta_2')
    parser.add_argument('--beta3', default=0.9, type=float, help='moving average coefficients beta_3')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--damping', default=0.01, type=float, help='damping factor for kfac and foof')
    parser.add_argument('--tcov', default=5, type=int, help='preconditioner update period for kfac and foof')
    parser.add_argument('--tinv', default=50, type=int, help='preconditioner inverse period for kfac and foof')
    parser.add_argument('--update', default=1, type=int, help='preconditioner update and inverse period for adaact')    
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler',
                       choices=['cosine', 'multistep'])

    return parser


def build_dataset(args):
    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False,
                                              num_workers=4)

    return train_loader, test_loader


def get_ckpt_name(model='resnet20', optimizer='sgd', lr=0.1, momentum=0.9, stat_decay=0.9,
                  beta1=0.9, beta2=0.999, beta3=0.9, eps=1e-8, weight_decay=5e-4,
                  damping=0.01, tcov=5, tinv=50, update=1,
                  run=0, lr_scheduler='cosine'):
    name = {
        'sgd': 'lr{}-momentum{}-wdecay{}-lr_sched{}-run{}'.format(lr, momentum, weight_decay,
                                                                  lr_scheduler, run),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-lr_sched{}-run{}'.format(lr, beta1, beta2,
                                                                         weight_decay, eps,
                                                                         lr_scheduler, run),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-lr_sched{}-run{}'.format(lr, beta1, beta2,
                                                                          weight_decay, eps,
                                                                          lr_scheduler, run),
        'adan': 'lr{}-betas{}-{}-{}-eps{}-wdecay{}-lr_sched{}-run{}'.format(
            lr, beta1, beta2, beta3, eps, weight_decay, lr_scheduler, run),
        'kfac': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, run),
        'foof': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, run),
        'adaact': 'lr{}-betas{}-{}-eps{}-wdecay{}-update{}-lr_sched{}-run{}'.format(
            lr, beta1, beta2, eps, weight_decay, update, lr_scheduler, run),
    }[optimizer]
    return '{}-{}-{}'.format(model, optimizer, name)


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet20': resnet20,
        'resnet32': resnet32,
        'resnet44': resnet44,
        'resnet56': resnet56,
        'resnet110': resnet110,
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgd':
        return SGD(model_params, args.lr, momentum=args.momentum,
                   weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                     weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adan':
        return Adan(model_params, args.lr, betas=(args.beta1, args.beta2, args.beta3),
                    weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'kfac':
        return KFAC(model_params, args.lr, momentum=args.momentum, stat_decay=args.stat_decay,
                    weight_decay=args.weight_decay, damping=args.damping,
                    Tcov=args.tcov, Tinv=args.tinv)
    elif args.optim == 'foof':
        return FOOF(model_params, args.lr, momentum=args.momentum, stat_decay=args.stat_decay,
                    weight_decay=args.weight_decay, damping=args.damping,
                    Tcov=args.tcov, Tinv=args.tinv)
    elif args.optim == 'adaact':
        return AdaAct(model_params, args.lr, betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay, eps=args.eps, update=args.update)
    else:
        print('Optimizer not found')


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              momentum=args.momentum, stat_decay=args.stat_decay,
                              beta1=args.beta1, beta2=args.beta2, beta3=args.beta3,
                              eps=args.eps, run=args.run,
                              weight_decay=args.weight_decay,
                              damping=args.damping, tcov=args.tcov, tinv=args.tinv,
                              update=args.update, lr_scheduler=args.lr_scheduler)
    print('ckpt_name')
    # load a checkpoint
    ckpt = load_checkpoint(ckpt_name)

    curve = os.path.join('curve', ckpt_name)
    curve = torch.load(curve)

    # build a network
    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    topk = 5

    eigvals, eigvecs = hess_scipy(net, topk, train_loader, criterion, device)

    # save eigenvalues
    with open('eigenvalues.pth', 'wb') as fout:
        pickle.dump(eigvals, fout)


if __name__ == '__main__':
    main()
