"""Train CIFAR100 with PyTorch."""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
from torch.optim import Adam, SGD, AdamW
#from optimizers.kfac import KFAC
from optimizers.kfac2 import KFAC
from optimizers.foof import FOOF
from optimizers.mac import MAC
from optimizers.smac import SMAC
from optimizers.adahessian import Adahessian
from optimizers.eva import Eva
from optimizers.lngd import LNGD


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--epoch', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet20', 'resnet32','resnet110','resnet', 'densenet', 'wrn'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=['sgd', 'adam', 'adamw', 'adan', 'kfac', 'foof', 'adahessian', 'eva', 'lngd',
                                 'mac', 'smac'])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for numerical stability')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--stat_decay', default=1e-4, type=float, help='stat decay')
    parser.add_argument('--beta1', default=0.9, type=float, help='moving average coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='moving average coefficients beta_2')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizers')
    parser.add_argument('--damping', default=0.01, type=float, help='damping factor')
    parser.add_argument('--mu', default=0.03, type=float, help='damping factor adaptive lr')
    parser.add_argument('--nu1', default=1e-5, type=float, help='damping lower threshold')
    parser.add_argument('--nu2', default=1e-2, type=float, help='damping upper threshold')
    parser.add_argument('--tcov', default=5, type=int, help='preconditioner update period')
    parser.add_argument('--tinv', default=50, type=int, help='preconditioner inverse period')
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
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, 
                                              num_workers=4, pin_memory=True, drop_last=True)

    return train_loader, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, momentum=0.9, stat_decay=0.9,
                  beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=5e-4,
                  damping=0.01, tcov=5, tinv=50, batchsize=128, mu=0.03, nu1=1e-5, nu2=1e-2,
                  epoch=200, run=0, lr_scheduler='cosine'):
    name = {
        'sgd': 'lr{}-momentum{}-wdecay{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, weight_decay, lr_scheduler, batchsize, epoch, run),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, beta1, beta2, weight_decay, eps, lr_scheduler, batchsize, epoch, run),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, beta1, beta2, weight_decay, eps, lr_scheduler, batchsize, epoch, run),
        #'kfac': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
        #    lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, batchsize, epoch, run),
        'foof': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, batchsize, epoch, run),
        'mac': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, batchsize, epoch, run),
        'smac': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, batchsize, epoch, run),
        'lngd': 'lr{}-momentum{}-stat_decay{}-mu{}-wdecay{}-tcov{}-tinv{}-nu1{}-nu2{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, stat_decay, mu, weight_decay, tcov, tinv, nu1, nu2, lr_scheduler, batchsize, epoch, run),
        'adahessian': 'lr{}-betas{}-{}-wdecay{}-eps{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, beta1, beta2, weight_decay, eps, lr_scheduler, batchsize, epoch, run),
        'eva': 'lr{}-momentum{}-wdecay{}-stat_decay{}-damping{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, weight_decay, stat_decay, damping, tcov, tinv, lr_scheduler, batchsize, epoch, run),
        'kfac': 'lr{}-momentum{}-wdecay{}-stat_decay{}-damping{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, weight_decay, stat_decay, damping, tcov, tinv, lr_scheduler, batchsize, epoch, run),
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
        'resnet110': resnet110,
        'resnet': ResNet34,
        'densenet': DenseNet121,
        'wrn': wrn_28_10,
        #'deit_tiny': DeiT32Tiny
        }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        #net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgd':
        return SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay, nesterov=True)
    elif args.optim == 'adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adanorm':
        return AdaNorm(model_params, args.lr, beta1=args.beta1, beta2=args.beta2,
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    #elif args.optim == 'kfac':
    #    return KFAC(model_params, args.lr, momentum=args.momentum, stat_decay=args.stat_decay,
    #                  weight_decay=args.weight_decay, damping=args.damping, Tcov=args.tcov, Tinv=args.tinv)
    elif args.optim == 'foof':
        return FOOF(model_params, args.lr, momentum=args.momentum, stat_decay=args.stat_decay,
                      weight_decay=args.weight_decay, damping=args.damping, Tcov=args.tcov, Tinv=args.tinv)
    elif args.optim == 'mac':
        return MAC(model_params, args.lr, args.momentum, stat_decay=args.stat_decay, damping=args.damping,
                   weight_decay=args.weight_decay, Tcov=args.tcov, Tinv=args.tinv)
    elif args.optim == 'smac':
        return SMAC(model_params, args.lr, args.momentum, stat_decay=args.stat_decay, damping=args.damping,
                    weight_decay=args.weight_decay, Tcov=args.tcov, Tinv=args.tinv)
    elif args.optim == 'lngd':
        return LNGD(model_params, args.lr, args.momentum, stat_decay=args.stat_decay,
                         mu=args.mu, weight_decay=args.weight_decay, Tcov=args.tcov, Tinv=args.tinv,
                        nu1=args.nu1, nu2=args.nu2)
    elif args.optim == 'adahessian':
        return Adahessian(model_params, args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optim == 'eva':
        return SGD(model_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'kfac':
        return SGD(model_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print('Optimizer not found')


def train(net, epoch, device, data_loader, optimizer, criterion, args, preconditioner):
    print('\nEpoch: %d' % epoch)
    net.train()
    tr_loss = 0
    correct = 0
    total = 0
        
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if args.optim in ['adahessian']:
            loss.backward(create_graph = True)
        else:
            loss.backward()
        if args.optim in ['eva', 'kfac'] and preconditioner is not None:
            preconditioner.step()
        optimizer.step()
        
        tr_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_loss = tr_loss / (batch_idx + 1)
    
    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)

    return accuracy, train_loss


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)

    return accuracy


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr, momentum=args.momentum,
                              stat_decay=args.stat_decay, beta1=args.beta1, beta2=args.beta2, eps=args.eps,
                              weight_decay=args.weight_decay, damping=args.damping, tcov=args.tcov, tinv=args.tinv,
                              mu=args.mu, nu1=args.nu1, nu2=args.nu2,
                              epoch=args.epoch, batchsize=args.batchsize, lr_scheduler=args.lr_scheduler, run=args.run,
                              )
    print('ckpt_name:', ckpt_name)
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

        curve = os.path.join('curve', ckpt_name)     
        curve = torch.load(curve)
        train_losses = curve['train_loss']
        train_accuracies = curve['train_acc']
        test_accuracies = curve['test_acc']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        execution_times = []

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())

    if args.optim in ['foof', 'lngd']:
        optimizer.model = net
    elif args.optim in ['mac', 'smac']:
        optimizer._configure(train_loader, net, device)

    preconditioner = None
    if args.optim in ['eva']:
        preconditioner = Eva(net, factor_decay=args.stat_decay, damping=args.damping,
                            fac_update_freq=args.tcov, kfac_update_freq=args.tinv)
    elif args.optim in ['kfac']:
        preconditioner = KFAC(net, factor_decay=args.stat_decay, damping=args.damping,
                            fac_update_freq=args.tcov, kfac_update_freq=args.tinv)
    
    # learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    elif args.lr_scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[50, 100, 150],
                                                   gamma=0.1)
    else:
        print('Learning rate scheduler not found')   

    tik = time.time()
    for epoch in range(start_epoch + 1, args.epoch):
        start = time.time()
        train_acc, train_loss = train(net, epoch, device, train_loader, optimizer, criterion, args, preconditioner)
        end = time.time()
        test_acc = test(net, device, test_loader, criterion)
        scheduler.step()
        execution_time = end - start
        print('Time: {}'.format(execution_time))

        if epoch == 0:
            MB = 1024.0 * 1024.0
            print('GPU Memory Usage: {}'.format(torch.cuda.max_memory_allocated() / MB))

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        execution_times.append(execution_time)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_loss': train_losses, 'train_acc': train_accuracies, 'test_acc': test_accuracies,
                    'time': execution_times}, os.path.join('curve', ckpt_name))
    tok = time.time()
    print('Total Time: {}'.format(tok-tik))


if __name__ == '__main__':
    main()