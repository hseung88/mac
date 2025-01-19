"""Train ImageNet32 with PyTorch."""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import os
import argparse
import time
from imagenet32_dataset import ImageNet32
from models import *
from torch.optim import Adam, SGD, AdamW
#from optimizers.kfac import KFAC
from optimizers.kfac2 import KFAC
from optimizers.foof import FOOF
from optimizers.adaact_v2 import AdaAct
from optimizers.adaact_one import AdaActR1
from optimizers.mac import MAC
from optimizers.smac import SMAC
from optimizers.sgdhess import SGDHess
from optimizers.adahessian import Adahessian
from optimizers.eva import Eva
#from optimizers.nysact_mod import NysAct
#from optimizers.shaper import Shaper

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--data_dir', default='./datasets', type=str)
    parser.add_argument('--epoch', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet32','resnet110','resnet50', 'densenet', 'wrn', 'deit_s'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=['sgd', 'adam', 'adamw', 'adan', 'kfac', 'foof', 'adaact', 'shaper', 'adaact_r1',
                                 'mac', 'smac', 'sgdhess', 'adahessian', 'eva', 'nysact'])
    parser.add_argument('--run_id', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for numerical stability')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--stat_decay', default=1e-4, type=float, help='stat decay')
    parser.add_argument('--beta1', default=0.9, type=float, help='moving average coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='moving average coefficients beta_2')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    #parser.add_argument('--rank', default=5, type=int, help='the number of subcolumns used in nysact')
    parser.add_argument('--damping', default=0.01, type=float, help='damping factor for kfac and foof')
    parser.add_argument('--tcov', default=5, type=int, help='preconditioner update period for kfac and foof')
    parser.add_argument('--tinv', default=50, type=int, help='preconditioner inverse period for kfac and foof')
    parser.add_argument('--update', default=1, type=int, help='preconditioner update and inverse period')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler',
                       choices=['cosine', 'multistep'])
    
    return parser

def build_dataset(args):
    if dist.get_rank() == 0:
        print('==> Preparing ImageNet32 data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load ImageNet32 dataset
    trainset = ImageNet32(root=args.data_dir, train=True, transform=transform_train)
    testset = ImageNet32(root=args.data_dir, train=False, transform=transform_test)

    # Distributed sampler ensures each GPU gets a different subset of data
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, sampler=train_sampler,
                                               num_workers=16, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, sampler=test_sampler,
                                              num_workers=16, pin_memory=True)

    return train_loader, test_loader



def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, momentum=0.9, stat_decay=0.9,
                  beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=5e-4, 
                  damping=0.01, tcov=5, tinv=50, update=1, batchsize=128,
                  epoch=200, run_id=0, lr_scheduler='cosine'):
    name = {
        'sgd': 'lr{}-momentum{}-wdecay{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, weight_decay, lr_scheduler, batchsize, epoch, run_id),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, beta1, beta2, weight_decay, eps, lr_scheduler, batchsize, epoch, run_id),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, beta1, beta2, weight_decay, eps, lr_scheduler, batchsize, epoch, run_id),
        #'kfac': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
        #    lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, batchsize, epoch, run),
        'foof': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, batchsize, epoch, run_id),
        'adaact': 'lr{}-betas{}-{}-eps{}-wdecay{}-update{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, beta1, beta2, eps, weight_decay, update, lr_scheduler, batchsize, epoch, run_id),
        'adaact_r1': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-update{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, update, lr_scheduler, batchsize, epoch, run_id),
        'mac': 'lr{}-momentum{}-damping{}-wdecay{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, damping, weight_decay, tinv, lr_scheduler, batchsize, epoch, run_id),
        'smac': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-update{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, update, lr_scheduler, batchsize, epoch, run_id),
        'sgdhess': 'lr{}-momentum{}-wdecay{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, weight_decay, lr_scheduler, batchsize, epoch, run_id),
        'adahessian': 'lr{}-betas{}-{}-wdecay{}-eps{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, beta1, beta2, weight_decay, eps, lr_scheduler, batchsize, epoch, run_id),
        'eva': 'lr{}-momentum{}-wdecay{}-stat_decay{}-damping{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, weight_decay, stat_decay, damping, tcov, tinv, lr_scheduler, batchsize, epoch, run_id),
        'kfac': 'lr{}-momentum{}-wdecay{}-stat_decay{}-damping{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, weight_decay, stat_decay, damping, tcov, tinv, lr_scheduler, batchsize, epoch, run_id),
        'shaper': 'lr{}-momentum{}-stat_decay{}-damping{}-wdecay{}-tcov{}-tinv{}-lr_sched{}-batchsize{}-epoch{}-run{}'.format(
            lr, momentum, stat_decay, damping, weight_decay, tcov, tinv, lr_scheduler, batchsize, epoch, run_id),
    }[optimizer]
    return '{}-{}-{}'.format(model, optimizer, name)


def load_checkpoint(ckpt_name):
    if dist.get_rank() == 0:
        print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    if dist.get_rank() == 0:
        print('==> Building model..')
    net = {
        'resnet32': resnet32,
        'resnet110': resnet110,
        'resnet50': ResNet50,
        'densenet': DenseNet121,
        'wrn': wrn_28_10,
        #'efficientnet': efficientnet_cifar,
        'deit_s': DeiT32
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
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
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
    elif args.optim == 'adaact':
        return AdaAct(model_params, args.lr, betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay, eps=args.eps, update=args.update)
    elif args.optim == 'adaact_r1':
        return AdaActR1(model_params, args.lr, args.momentum, stat_decay=args.stat_decay, 
                         damping=args.damping, weight_decay=args.weight_decay, update_freq=args.update)
    elif args.optim == 'mac':
        return MAC(model_params, args.lr, args.momentum, 
                        damping=args.damping, weight_decay=args.weight_decay, Tinv=args.tinv)
    elif args.optim == 'smac':
        return SMAC(model_params, args.lr, args.momentum, stat_decay=args.stat_decay, 
                         damping=args.damping, weight_decay=args.weight_decay, update_freq=args.update)
    elif args.optim == 'sgdhess':
        return SGDHess(model_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adahessian':
        return Adahessian(model_params, args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optim == 'eva':
        return SGD(model_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'kfac':
        return SGD(model_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'shaper':
        return Shaper(model_params, args.lr, momentum=args.momentum, stat_decay=args.stat_decay,
                      weight_decay=args.weight_decay, damping=args.damping, Tcov=args.tcov, Tinv=args.tinv)
    else:
        print('Optimizer not found')

def train(net, epoch, device, data_loader, optimizer, criterion, args, preconditioner):
    if dist.get_rank() == 0:
        print(f'\nEpoch: {epoch}')
    net.train()
    tr_loss = 0.0
    correct = 0
    total = 0
    
    # Ensure DistributedSampler is shuffled
    data_loader.sampler.set_epoch(epoch)
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if args.optim in ['sgdhess', 'adahessian']:
            loss.backward(create_graph = True)
        else:
            loss.backward()
        if args.optim in ['eva', 'kfac'] and preconditioner is not None:
        #if args.optim in ['eva'] and preconditioner is not None:
            preconditioner.step()
        optimizer.step()

        tr_loss += loss.item() * inputs.size(0)  # Accumulate loss (sum over the batch)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Average the metrics across all GPUs
    tr_loss = torch.tensor(tr_loss, dtype=torch.float32, device=device)
    correct = torch.tensor(correct, dtype=torch.float32, device=device)
    total = torch.tensor(total, dtype=torch.float32, device=device)

    # All-reduce across all processes to aggregate results
    dist.all_reduce(tr_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    train_loss = tr_loss.item() / total.item()
    accuracy = 100. * correct.item() / total.item()

    # Print only in the master process (rank 0)
    if dist.get_rank() == 0:
        print(f'Train acc: {accuracy:.3f}, Train loss: {train_loss:.4f}')

    return accuracy, train_loss


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)  # Accumulate loss (sum over the batch)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Average the metrics across all GPUs
    test_loss = torch.tensor(test_loss, dtype=torch.float32, device=device)
    correct = torch.tensor(correct, dtype=torch.float32, device=device)
    total = torch.tensor(total, dtype=torch.float32, device=device)

    # All-reduce across all processes to aggregate results
    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    test_loss = test_loss.item() / total.item()
    accuracy = 100. * correct.item() / total.item()

    # Print only in the master process (rank 0)
    if dist.get_rank() == 0:
        print(f'Test acc: {accuracy:.3f}, Test loss: {test_loss:.4f}')

    return accuracy


def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    
    parser = get_parser()
    args = parser.parse_args()

    # Get the local rank from the environment variable, not from args
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    train_loader, test_loader = build_dataset(args)


    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              momentum=args.momentum, stat_decay=args.stat_decay,
                              beta1=args.beta1, beta2=args.beta2,
                              eps = args.eps, run_id=args.run_id,
                              weight_decay = args.weight_decay,
                              damping=args.damping, tcov=args.tcov, tinv=args.tinv,
                              epoch=args.epoch, batchsize=args.batchsize,
                              update=args.update, lr_scheduler=args.lr_scheduler,
                              )
    if dist.get_rank() == 0:
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
    # Wrap model in DistributedDataParallel
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())
    
    #if args.optim in ['foof', 'adaact', 'nysact', 'shaper', 'kfac']:
    if args.optim in ['foof', 'adaact', 'nysact_g', 'nysact_s', 'nysact_ls', 'shaper', 'els']:
        optimizer.model = net
    elif args.optim in ['mac','smac', 'adaact_r1']:
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
    best_acc = 0  # Only save the best model in rank 0
    
    for epoch in range(start_epoch + 1, args.epoch):
        start = time.time()
        train_acc, train_loss = train(net, epoch, device, train_loader, optimizer, criterion, args, preconditioner)
        end = time.time()
        test_acc = test(net, device, test_loader, criterion)
        scheduler.step()
        execution_time = end - start
        # Only rank 0 logs time and saves the model
        if dist.get_rank() == 0:
            print(f'Time: {execution_time}')
            
            # Save checkpoint if the test accuracy is better
            if test_acc > best_acc:
                if dist.get_rank() == 0:
                    print('Saving model...')
                state = {
                    'net': net.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, os.path.join('checkpoint', f'best_model_epoch_{epoch}.pth'))
                best_acc = test_acc
            
            #MB = 1024.0 * 1024.0
            #print(torch.cuda.max_memory_allocated() / MB)
    
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            test_accuracies.append(test_acc)
            execution_times.append(execution_time)
            if not os.path.isdir('curve'):
                os.mkdir('curve')
            torch.save({'train_loss': train_losses, 'train_acc': train_accuracies, 'test_acc': test_accuracies, 'time': execution_times}, 
                   os.path.join('curve', ckpt_name))
    tok = time.time()
    if dist.get_rank() == 0:
        print(f'Total Time: {tok - tik}')

    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()