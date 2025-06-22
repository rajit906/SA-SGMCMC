#!/usr/bin/env python3
"""
SGULA CIFAR10 Training with metrics logging.
Modified from SGHMC to use BAOAB splitting integrator.
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
import config as cf

# project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
from models.resnet import ResNet18

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description='SGULA CIFAR10 Training')
    parser.add_argument('--dir', default='cifar/ckpt_sgula', help='path to save checkpoints')
    parser.add_argument('--experiment_dir', default=None, help='Optional Path for temporary tests')
    parser.add_argument('--data_path', default='cifar/data', help='path to datasets')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='train batch size')
    parser.add_argument('--sampler_type', required=True, choices=['sgula', 'sa-sgula'])
    parser.add_argument('--gamma', type=float, default=0.1, help='friction coefficient')
    ### These are for SA-SGULA
    parser.add_argument('--alpha', default=50., type=float)
    parser.add_argument('--lr_init', default=0.1, type=float)
    parser.add_argument('--m', default=0.2, type=float)
    parser.add_argument('--M', default=2.0, type=float)
    parser.add_argument('--r', default=0.5, type=float)
    parser.add_argument('--omega', default=1., type=float)
    ###
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
    def make_model(device):
        net = ResNet18().to(device)
        return net

    net = make_model(device) 
    if use_cuda:
        cudnn.benchmark = True
        cudnn.deterministic = True

    # initial learning rate
    lr_0 = args.lr_init
    weight_decay = 5e-4
    zeta = torch.tensor(0.)
    
    # criterion and optimizer
    datasize = len(trainset)
    criterion = nn.CrossEntropyLoss()
    T = args.epochs * (datasize // args.batch_size + 1)
    num_batch = datasize // args.batch_size + 1

    # metrics storage
    metrics = {'train': {'psis': []}, 'test': {}}

    # Initialize momentum buffers
    for p in net.parameters():
        p.buf = torch.zeros_like(p.data)

    # learning rate scheduler
    def adjust_learning_rate(epoch, batch_idx, gradnorm, zeta):
        if args.sampler_type == 'sa-sgula':
            exptau = torch.exp(-torch.tensor(args.alpha * lr_0))
            zeta = exptau * zeta + (1 - exptau) * gradnorm / args.alpha
            zeta_r = zeta ** args.r
            lr = lr_0 * args.m * (zeta_r + args.M) / (zeta_r + args.m)
            lr = lr.item()
        elif args.sampler_type == 'sgula':
            if args.experiment_dir is not None:
                lr = float(args.experiment_dir)
            else:
                lr = lr_0
        return lr, zeta

    def compute_gradnorm():
        """Compute gradient norm for adaptive step size"""
        gradnorm = 0.
        for p in net.parameters():
            if p.grad is not None:
                gradnorm += torch.sum(p.grad.data ** 2)
        gradnorm *= 1 / (datasize * args.omega)
        return gradnorm

    def fill_gradients(inputs, targets):
        """Compute gradients for current batch"""
        net.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)  # No scaling!
        loss.backward()
        
        # Add weight decay to gradients
        for p in net.parameters():
            p.grad.data.add_(p.data, alpha=weight_decay)
        
        return outputs, loss

    def update_params_BAOA(lr, epoch):
        """Performs B-A-O-A steps using stored gradients"""
        # Compute coefficients
        a = np.exp(-args.gamma * lr)
        sqrt_aT = np.sqrt((1 - a**2) * args.temperature)
        
        for p in net.parameters():
            # B-step (first half) - using stored gradients
            p.buf.add_(p.grad.data, alpha=-0.5*lr)
            
            # A-step (first half)
            p.data.add_(p.buf, alpha=0.5*lr)
            
            # O-step
            eps = torch.randn(p.size()).to(device, non_blocking=True)
            p.buf.mul_(a)
            if epoch >= 150:  # Add noise only after burn-in
                p.buf.add_(eps, alpha=sqrt_aT)
            
            # A-step (second half)
            p.data.add_(p.buf, alpha=0.5*lr)

    def update_params_B(lr):
        """Performs final B-step with new gradients"""
        for p in net.parameters():
            # B-step (second half)
            p.buf.add_(p.grad.data, alpha=-0.5*lr)

    # Training and Testing functions
    def train(epoch, zeta):
        net.train()
        # init epoch metrics
        metrics['train'][epoch] = {'lr': [], 'loss': None, 'acc': None, 'zeta': []}
        epoch_loss = epoch_correct = epoch_total = 0

        # Convert to iterator for manual batch handling
        train_iter = iter(trainloader)
        
        # For first iteration of first epoch, compute initial gradients
        if epoch == 0:
            try:
                first_inputs, first_targets = next(train_iter)
                first_inputs, first_targets = first_inputs.to(device), first_targets.to(device)
                outputs, loss = fill_gradients(first_inputs, first_targets)
                
                # Initialize zeta if using SA-SGULA
                if args.sampler_type == 'sa-sgula':
                    gradnorm = compute_gradnorm()
                    zeta = gradnorm
                
                # Store metrics for first batch
                preds = outputs.argmax(dim=1)
                correct = (preds == first_targets).sum().item()
                total = first_targets.size(0)
                epoch_loss += loss.item() * total
                epoch_correct += correct
                epoch_total += total
            except StopIteration:
                pass

        batch_idx = 0
        while True:
            try:
                # Get current batch
                inputs, targets = next(train_iter)
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Compute gradient norm for adaptive step size (using current gradients)
                gradnorm = 0.
                if args.sampler_type == 'sa-sgula':
                    gradnorm = compute_gradnorm()
                    
                if zeta is not None:
                    metrics['train'][epoch]['zeta'].append(zeta.item())

                lr, zeta = adjust_learning_rate(epoch, batch_idx, gradnorm, zeta)
                
                # BAOAB: B-A-O-A using current gradients, then compute new gradients for next B
                update_params_BAOA(lr, epoch)
                
                # Try to get next batch for final B step
                try:
                    next_inputs, next_targets = next(train_iter)
                    next_inputs, next_targets = next_inputs.to(device), next_targets.to(device)
                    
                    # Compute gradients with next batch
                    outputs, loss = fill_gradients(next_inputs, next_targets)
                    
                    # Final B step with next batch gradients
                    update_params_B(lr)
                    
                    # Store metrics for next batch
                    preds = outputs.argmax(dim=1)
                    correct = (preds == next_targets).sum().item()
                    total = next_targets.size(0)
                    epoch_loss += loss.item() * total
                    epoch_correct += correct
                    epoch_total += total
                    metrics['train'][epoch]['lr'].append(lr)
                    
                except StopIteration:
                    # No next batch, use current batch for final B step
                    outputs, loss = fill_gradients(inputs, targets)
                    update_params_B(lr)
                    
                    # Store metrics for current batch
                    preds = outputs.argmax(dim=1)
                    correct = (preds == targets).sum().item()
                    total = targets.size(0)
                    epoch_loss += loss.item() * total
                    epoch_correct += correct
                    epoch_total += total
                    metrics['train'][epoch]['lr'].append(lr)
                    break
                
                batch_idx += 1
                
            except StopIteration:
                break

        metrics['train'][epoch]['loss'] = epoch_loss / epoch_total
        metrics['train'][epoch]['acc'] = epoch_correct / epoch_total
        return zeta

    def test(epoch):
        net.eval()
        metrics['test'][epoch] = {'lr': [], 'loss': None, 'acc': None}
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
        print('Loss: ', test_loss / total, "Acc: ", correct / total)
        metrics['test'][epoch]['loss'] = test_loss / total
        metrics['test'][epoch]['acc'] = correct / total

    mt = 1
    for epoch in tqdm(range(args.epochs)):
        zeta = train(epoch, zeta)
        test(epoch)
        
        # Print mean learning rate for this epoch
        if metrics['train'][epoch]['lr']:
            mean_lr = np.mean(metrics['train'][epoch]['lr'])
            print(f"Epoch {epoch}: Mean LR = {mean_lr:.6f}")
        
        cp_path = create_dir(args.dir, args.experiment_dir, args.sampler_type)
        if epoch > 150 and epoch % 3 == 0:
            if args.sampler_type == 'sgula':
                model_path = os.path.join(cp_path, 'S' + str(args.seed))
                save_model(net, model_path, mt, device)
                mt += 1
            elif args.sampler_type == 'sa-sgula':
                model_path = os.path.join(cp_path, 'S' + str(args.seed), 
                    f'lr{args.lr_init}_a{args.alpha}_m{args.m}_M{args.M}_r{args.r}_O{args.omega}')
                save_model(net, model_path, mt, device)
                psi = compute_psi(zeta, args.r, args.m, args.M)
                metrics['train']['psis'].append(psi) 
                mt += 1

    with open(os.path.join(model_path, 'train_metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved training metrics to {os.path.join(model_path, 'train_metrics.json')}")
    del net, trainloader, testloader, criterion
    torch.cuda.empty_cache()
    gc.collect()