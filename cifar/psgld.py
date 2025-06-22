#!/usr/bin/env python3
"""
Preconditioned SG-MCMC CIFAR10 Training with metrics logging.
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
from utils import save_model, create_dir
import config as cf

# project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
from models.resnet import ResNet18

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description='pSGLD CIFAR10 Training')
    parser.add_argument('--dir', default='cifar/ckpt_sgld', help='path to save checkpoints')
    parser.add_argument('--data_path', default='cifar/data', help='path to datasets')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='train batch size')
    parser.add_argument('--lr_init', default=1e-4, type=float, help='learning rate')
    ### Preconditioner parameters
    parser.add_argument('--alpha', default=0.99, type=float, help='RMSprop alpha parameter')
    parser.add_argument('--eps', default=1e-8, type=float, help='RMSprop eps parameter')
    ###
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--temperature', type=float, default=1./50000, help='temperature for Langevin noise')
    return parser.parse_args()


class PreconditionedSGLD(optim.Optimizer):
    """
    Preconditioned SGLD using PyTorch RMSprop-style preconditioning + Langevin noise
    """
    def __init__(self, params, lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0, 
                 temperature=1./50000, addnoise=True):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                       temperature=temperature, addnoise=addnoise)
        super(PreconditionedSGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]

                # State Initialization (exactly like PyTorch RMSprop)
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                # Update biased second raw moment estimate (exactly like PyTorch)
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                # Compute denominator (exactly like PyTorch)
                avg = square_avg.sqrt().add_(group['eps'])

                if group['addnoise']:
                    # Add Langevin noise scaled by preconditioner
                    noise_std = torch.sqrt(2.0 * group['lr'] * group['temperature'] / avg)
                    noise = torch.randn_like(p.data) * noise_std
                    
                    # SGLD update: p = p - lr * (0.5 * grad/avg + noise)
                    p.data.addcdiv_(grad, avg, value=-0.5 * group['lr']).add_(noise, alpha=-group['lr'])
                else:
                    # Regular RMSprop update: p = p - lr * grad/avg
                    p.data.addcdiv_(grad, avg, value=-group['lr'])

        return loss


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
        transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
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
    net = ResNet18().to(device) 
    if use_cuda:
        cudnn.benchmark = True
        cudnn.deterministic = True

    # initial learning rate
    lr_0 = args.lr_init
    
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = PreconditionedSGLD(net.parameters(), lr=lr_0, alpha=args.alpha,
                                  eps=args.eps, weight_decay=5e-4, 
                                  temperature=args.temperature, addnoise=False)

    # metrics storage
    metrics = {'train': {}, 'test': {}}

    def train(epoch):
        net.train()
        metrics['train'][epoch] = {'lr': [], 'loss': None, 'acc': None}
        epoch_loss = epoch_correct = epoch_total = 0
        
        # Enable noise after epoch 150
        add_noise = (epoch >= 150)
        for group in optimizer.param_groups:
            group['addnoise'] = add_noise
            
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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
            metrics['train'][epoch]['lr'].append(lr_0)
            
        metrics['train'][epoch]['loss'] = epoch_loss / epoch_total
        metrics['train'][epoch]['acc'] = epoch_correct / epoch_total

    def test(epoch):
        net.eval()
        metrics['test'][epoch] = {'loss': None, 'acc': None}
        test_loss = correct = total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            test_loss += loss.item() * targets.size(0)
        print('Loss: ', test_loss / total, "Acc: ", correct / total, "LR:", lr_0)
        metrics['test'][epoch]['loss'] = test_loss / total
        metrics['test'][epoch]['acc'] = correct / total

    # -- main loop --
    mt = 1
    for epoch in tqdm(range(args.epochs)):
        train(epoch)
        test(epoch)
        
        cp_path = create_dir(args.dir, str(lr_0), 'psgld')
        if epoch > 150 and epoch % 3 == 0:
            model_path = os.path.join(cp_path, 'S' + str(args.seed))
            save_model(net, model_path, mt, device)
            mt += 1

    # Save metrics
    model_path = os.path.join(cp_path, 'S' + str(args.seed))
    with open(os.path.join(model_path, 'train_metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved training metrics to {os.path.join(model_path, 'train_metrics.json')}")

    del net, trainloader, testloader, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()