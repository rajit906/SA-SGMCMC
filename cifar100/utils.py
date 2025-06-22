import os
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def custom_schedule(x, init_val, target_val, plateau_step, decay_rate=0.0):
    """
    Two-phase schedule:
      1) Fast exp decay from init_val @ x=1 to target_val @ x=plateau_step
      2) Plateau or slow exp decay thereafter (decay_rate)
    """
    if x < 1:
        raise ValueError("Step x must be >= 1")
    # Solve for exponent a: init_val * (plateau_step)^(-a) = target_val
    a = -np.log(target_val / init_val) / np.log(plateau_step)
    if x <= plateau_step:
        return init_val * (x ** -a)
    else:
        return target_val * ((x / plateau_step) ** -decay_rate)

def create_dir(dir, experiment_dir, sampler_type):
    cp_path = os.path.join(dir, sampler_type)
    if experiment_dir is not None:
        cp_path = os.path.join(cp_path, experiment_dir)
    return cp_path

def save_model(net, cp_path, mt, device):
    net.cpu()
    model_path = os.path.join(cp_path, f'{mt}.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(net.state_dict(), model_path)
    net.to(device)
        
def compute_psi(zeta, r, m, M):
    zeta_r = zeta ** r
    psi = m * (zeta_r + M) / (zeta_r + m)
    return psi.item()

def predict(dataloader, models, device, psis):
    """
    Runs ensemble predictions and computes (on device):
      - predictive probabilities (all_p)
      - true labels          (all_y)
      - predictive entropy   (all_H)
      - mutual information   (all_MI)
    Returns all arrays as NumPy on CPU at the end.
    """
    # buffers (torch) on device
    all_p = []
    all_y = []
    all_H = []
    all_MI = []
    total_samples = 0
    ws = torch.tensor(psis, device=device)
    wsum = ws.sum()

    with torch.no_grad():
        for _, (x, y) in enumerate(dataloader):
            B = y.size(0)
            total_samples += B
            x = x.to(device)
            y = y.to(device)
            # stack logits from each model: shape (M, B, C)
            logits = torch.stack([m(x) for m in models], dim=0)
            # compute predictive probabilities
            ps = torch.softmax(logits, dim=2)        # (M,B,C)
            mean_p = (ws[:,None,None] * ps).sum(0) / wsum # ps.mean(dim=0)
            # entropy H[ŷ] = -∑ mean_p * log(mean_p)
            H = -(mean_p * mean_p.log()).sum(dim=1)  # (B,)
            # cond_ent = - (ps * ps.log()).sum(dim=2).mean(dim=0)  # expected entropy E[H[y|w]] = mean over models of (-∑ p*log p)
            # H_i = -(p_i * log p_i).sum(1) per-model entropies
            Hi = -(ps * ps.log()).sum(2)    # shape (M,B)
            # weighted expected entropy
            cond_ent = (ws[:,None] * Hi).sum(0) / wsum
            # mutual information MI = H[ŷ] − E_w[H[y|w]]
            MI = H - cond_ent   # (B,)
            # collect
            all_p.append(mean_p)
            all_y.append(y)
            all_H.append(H)
            all_MI.append(MI)

    # concatenate everything and move to CPU/NumPy
    all_p   = torch.cat(all_p,  dim=0).cpu().numpy()
    all_y   = torch.cat(all_y,  dim=0).cpu().numpy()
    all_H   = torch.cat(all_H,  dim=0).cpu().numpy()
    all_MI  = torch.cat(all_MI, dim=0).cpu().numpy()

    return all_p, all_y, all_H, all_MI


