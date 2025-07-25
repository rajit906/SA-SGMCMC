#!/usr/bin/env python3
"""
Combined evaluation script: loads an ensemble of CIFAR-100 models and computes post-training metrics.
Usage:
  python cifar_ensemble.py --dir PATH_TO_MODELS --data_path PATH_TO_DATA [--device_id 0] [--batch_size 128] [--seed 1]
Outputs metrics.json in the models directory.
"""
import os
import sys
import gc
import glob
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config as cf
from sklearn.metrics import accuracy_score, log_loss
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.resnet import ResNet18
from cifar.utils import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Path to model checkpoints')
    parser.add_argument('--data_path', type=str, default='cifar100/data')
    parser.add_argument('--ensemble_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_corruption', action='store_true')
    args = parser.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # CIFAR-100 test loader (from existing script)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100'])
    ])
    testset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Load ensemble checkpoints
    model_paths = sorted(glob.glob(os.path.join(args.dir, '*.pt')))
    #train_metrics = json.load(open(os.path.join(args.dir, 'train_metrics.json')))

    assert model_paths, f"No checkpoints found in {args.dir}"
    ensemble = []
    for path in model_paths:
        net = ResNet18(num_classes=100).to(device).eval()
        net.load_state_dict(torch.load(path, map_location=device))
        ensemble.append(net)
    ensemble = ensemble[-args.ensemble_size:]
    ensemble_size = len(ensemble)
    
    #if 'psis' in train_metrics['train'] and len(train_metrics['train']['psis']) > 0 and False:
    #    psis = train_metrics['train']['psis']
    #    psis = psis[-ensemble_size:]
    #else:
    psis = [1/ensemble_size for _ in range(ensemble_size)]

    # Run predictions and compute uncertainties
    probs, y, H, MI = predict(testloader, ensemble, device, psis)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y, preds)
    nll = log_loss(y, probs)
    print(f"Accuracy {acc:.4f}, NLL {nll:.4f}")
    # Save all metrics to JSON
    out = {
        'accuracy': acc,
        'nll': nll,
        'probs': probs.tolist(),
        'y': y.tolist(),
        'entropy': H.tolist(),
        'mutual_information': MI.tolist(),
                }

    
    json_path = os.path.join(args.dir, f'eval_metrics_{ensemble_size}.json')
    with open(json_path, 'w') as f:
        json.dump(out, f)
    print(f"Saved metrics to {json_path}")
    del testloader
    for net in ensemble:
        del net

if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()
    gc.collect()

