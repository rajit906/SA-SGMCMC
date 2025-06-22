#!/usr/bin/env python3
"""
SG-MCMC CIFAR100 Training with metrics logging.
"""
from __future__ import print_function
import sys
import os
import gc
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import save_model, compute_psi

# project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
from models.resnet import ResNet18

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Deep Ensemble CIFAR-100 Training')
    parser.add_argument('--data_path', default='cifar/data', help='path to datasets')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='train batch size')
    ### These are for SA-SGLD
    parser.add_argument('--alpha', default=50., type=float)
    parser.add_argument('--lr_init', default=0.1, type=float)  # This is for SGLD as well
    parser.add_argument('--m', default=0.02, type=float)
    parser.add_argument('--M', default=2., type=float)
    parser.add_argument('--r', default=0.25, type=float)
    ###
    return parser.parse_args()

# learning rate scheduler
def adjust_learning_rate(gradnorm, zeta):
    exptau = torch.exp(-torch.tensor(args.alpha * lr_0))
    zeta = exptau * zeta + (1 - exptau) * gradnorm / args.alpha
    zeta_r = zeta ** args.r
    lr = lr_0 * args.m * (zeta_r + args.M) / (zeta_r + args.m)
    lr = lr.item()
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr, zeta

def train(epoch, zeta):
    net.train()
    metrics['train'][epoch] = {'lr': [], 'loss': None, 'acc': None, 'zeta': []}
    epoch_loss = epoch_correct = epoch_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        gradnorm = 0
        for p in list(net.parameters()):
            if p.grad is not None:
                gradnorm += torch.sum(p.grad.data ** 2)
        gradnorm *= 1 / datasize
        if zeta is not None:
            metrics['train'][epoch]['zeta'].append(zeta.item())
        lr, zeta = adjust_learning_rate(gradnorm, zeta)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        epoch_loss += loss.item() * total
        epoch_correct += correct
        epoch_total += total
        metrics['train'][epoch]['lr'].append(lr)
    metrics['train'][epoch]['loss'] = epoch_loss / epoch_total
    metrics['train'][epoch]['acc'] = epoch_correct / epoch_total
    return zeta

def test(epoch):
    net.eval()
    metrics['test'][epoch] = {'lr': [], 'loss': None, 'acc': None}
    test_loss = correct = total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        lr = optimizer.param_groups[0]['lr']
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        test_loss += loss.item() * targets.size(0)
        metrics['test'][epoch]['lr'].append(lr)
    print('Loss: ', test_loss / total, "Acc: ", correct / total)
    metrics['test'][epoch]['loss'] = test_loss / total
    metrics['test'][epoch]['acc'] = correct / total


if __name__ == "__main__":
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda' if use_cuda else 'cpu')
    for seed in range(8+1,16+1):
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()
        )
        testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # build model
        def make_model(device):
            net = ResNet18(num_classes=100).to(device)
            return net

        net = make_model(device) 
        if use_cuda:
            cudnn.benchmark = True
            cudnn.deterministic = True

        # initial learning rate
        lr_0 = args.lr_init
        
        # criterion and optimizer
        datasize = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr_0, weight_decay=5e-4)
        T = args.epochs * (datasize // args.batch_size + 1)
        num_batch = datasize // args.batch_size + 1

        # metrics storage
        metrics = {'train': {'psis': []}, 'test': {}}

        # -- main loop --
        zeta = torch.tensor(0.)
        for epoch in tqdm(range(args.epochs)):
            zeta = train(epoch, zeta)
            test(epoch)

        cp_path = 'cifar100/ckpt_desad'
        save_model(net, cp_path, seed, device)
        psi = compute_psi(zeta, args.r, args.m, args.M)
        metrics['train']['psis'].append(psi) 

        with open(os.path.join(cp_path, f'train_metrics_{seed}.json'), 'w') as fp:
            json.dump(metrics, fp, indent=2)
        del net, trainloader, testloader, optimizer, criterion
        torch.cuda.empty_cache()
        gc.collect()