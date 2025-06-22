#!/usr/bin/env python3
"""
SG-MCMC CIFAR10 Training with metrics logging.
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
from utils import save_model, create_dir, compute_psi, custom_schedule

# project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
from models.resnet import ResNet18

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description='SGLM CIFAR10 Training')
    parser.add_argument('--dir', default='cifar/ckpt_sglm', help='path to save checkpoints')
    parser.add_argument('--experiment_dir', default=None, help='Optional Path for temporary tests')
    parser.add_argument('--data_path', default='cifar/data', help='path to datasets')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='train batch size')
    parser.add_argument('--sampler_type', required = True, choices = ['sglm', 'sa-sglm'])
    parser.add_argument('--lr_init', default=0.1, type=float)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--temperature', type=float, default=1./50000, help='temperature')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda' if use_cuda else 'cpu')

    # prepare data
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
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()
    )
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()
    )

    # build model
    def make_model(device):
        net = ResNet18().to(device)
        return net

    net = make_model(device) 
    if use_cuda:
        cudnn.benchmark = True
        cudnn.deterministic = True

    # initial learning rate
    lr_0 = args.lr_init
    if args.sampler_type == 'sa-sglm':
        zeta = torch.tensor(0.)
    else:
        zeta = None
    
    # criterion and optimizer
    datasize = len(trainset)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr_0, weight_decay=5e-4)
    T = args.epochs * (datasize // args.batch_size + 1)
    num_batch = datasize // args.batch_size + 1

    # metrics storage
    metrics = {'train': {}, 'test': {}}

    # learning rate scheduler
    def adjust_learning_rate(epoch, batch_idx, gradnorm, zeta):
        rcounter = epoch * num_batch + batch_idx + 1
        if args.sampler_type == 'sa-sglm':
            exptau = torch.exp(-torch.tensor(50. * lr_0))
            zeta = exptau * zeta + (1 - exptau) * gradnorm / 50
            zeta_r = zeta ** 0.25
            lr = lr_0 * 0.02 * (zeta_r + 2) / (zeta_r + 0.02)
            lr = lr.item()
        else:
            if args.experiment_dir == 'higher_schedule_mean':
                lr = 0.063
            elif args.experiment_dir == 'higher_schedule_min':
                lr = 0.03
            elif args.experiment_dir == 'higher_schedule_max':
                lr = 0.1
            else:
                if epoch < 150:
                    lr = lr_0 * (rcounter) ** (-0.2)
                else:
                    rc_new = (epoch - 150) * num_batch + batch_idx + 1
                    lr = lr_0/10 * (rc_new) ** -0.5
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr, zeta

    def noise_loss(lr):
        noise_std = np.sqrt(2 / lr)
        loss = 0.0
        for var in net.parameters():
            noise = torch.normal(torch.zeros_like(var), noise_std).to(device)
            loss += torch.sum(var * noise)
        return loss

    def train(epoch, prev_noise, zeta):
        net.train()
        metrics['train'][epoch] = {'lr': [], 'loss': None, 'acc': None}
        epoch_loss = epoch_correct = epoch_total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            gradnorm = 0
            if args.sampler_type == 'sa-sglm':
                for p in list(net.parameters()):
                    if p.grad is not None:
                        gradnorm += torch.sum(p.grad.data ** 2)
                gradnorm *= 1 / datasize

            lr, zeta = adjust_learning_rate(epoch, batch_idx, gradnorm, zeta)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if epoch >= 150:
                curr_noise = noise_loss(lr) * np.sqrt(args.temperature / datasize)
                if prev_noise is not None:
                    noise_term = 0.5 * (prev_noise + curr_noise)
                    (loss + noise_term).backward(retain_graph=True)
                else:
                    noise_term = curr_noise
                    (loss + noise_term).backward()
                prev_noise = curr_noise.detach_()
            else:
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
        return prev_noise, zeta

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

    # -- main loop --
    mt = 1
    prev_noise = None
    for epoch in tqdm(range(args.epochs)):
        prev_noise, zeta = train(epoch, prev_noise, zeta)
        test(epoch)
        cp_path = create_dir(args.dir, args.experiment_dir, args.sampler_type)
        if epoch > 150 and epoch % 3 == 0:
            cp_path = os.path.join(cp_path, 'S' + str(args.seed))
            save_model(net, cp_path, mt, device)
            mt += 1

    with open(os.path.join(cp_path, 'train_metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved training metrics to {os.path.join(cp_path, 'train_metrics.json')}")
    del net, trainloader, testloader, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()


