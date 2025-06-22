import os
import gc
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'font.family': 'serif',
    'mathtext.fontset': 'cm'  # Computer Modern font for math symbols
})
from sklearn.metrics import roc_auc_score
import seaborn as sns

def evaluate_and_plot(
    model_dirs,
    ood_dataset_name='stl10',
    ensemble_size=16,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #fig_loss, ax_loss = plt.subplots(figsize=(8,6))
    fig_acc, ax_acc = plt.subplots(figsize=(8,6))
    mi_matrices = {}
    lr_data = {}
    zeta_data = {}
    entropy_data = {}
    calib_data = {}
    acc_values = {}
    last_pred, last_y_true = None, None
    for label, dir_path in model_dirs.items():
        lr_entry, zeta_entry = process_train_metrics(dir_path, label, ax_acc)
        lr_data[label] = lr_entry
        zeta_data[label] = zeta_entry
        mi_mat, entropy, calib, ood_info, pred, y_true, acc, nll = process_eval_metrics(dir_path, label, ood_dataset_name, ensemble_size=ensemble_size)
        mi_matrices[label] = mi_mat
        entropy_data[label] = entropy
        calib_data[label] = calib
        last_pred, last_y_true = pred, y_true
        acc_values[label] = acc
        print(f"{label} CIFAR-10: Acc={acc:.4f}, NLL={nll:.4f}")
        
        if ood_info:
            auroc, acc_ood, nll_ood = ood_info
            print(f"{label} {ood_dataset_name}: AUROC={auroc:.3f}, Acc={acc_ood:.4f}, NLL={nll_ood:.4f}")
    plot_stepsize_schedule(lr_data, zeta_data, acc_values)
    # finalize_loss_acc_plots(ax_acc)
    # if ood_dataset_name:
    #     plot_mi_matrices(mi_matrices)
    # if last_pred is not None and last_y_true is not None:
    #     plot_entropy_ecdfs(entropy_data, last_pred, last_y_true)
    
    for label, (correct, prob_correct) in calib_data.items():
        ece = compute_ece(correct.astype(int), prob_correct)
        print(f"{label} ECE: {ece:.4f}")
    
    plt.tight_layout()
    plt.show()

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = y_true.shape[0]
    
    for i in range(n_bins):
        # Include lower bound (>=)
        mask = (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            bin_weight = mask.sum() / N
            ece += bin_weight * abs(acc - conf)
            
    return ece

def compute_ace(y_true, y_prob, n_bins=10):
    """
    Compute Adaptive Calibration Error (ACE)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities (confidence scores)
        n_bins: Number of quantile-based bins
    
    Returns:
        ace: Adaptive Calibration Error
    """
    # Sort predictions and labels by confidence
    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]
    
    # Split into approximately equal-sized bins
    bins_y_true = np.array_split(y_true_sorted, n_bins)
    bins_y_prob = np.array_split(y_prob_sorted, n_bins)
    
    ace = 0.0
    for b_true, b_prob in zip(bins_y_true, bins_y_prob):
        if len(b_true) == 0:
            continue
        acc = np.mean(b_true)
        conf = np.mean(b_prob)
        ace += np.abs(acc - conf)
    
    return ace / n_bins

def process_train_metrics(dir_path, label, ax_acc):
    train_metrics_path = os.path.join(dir_path, 'train_metrics.json')
    with open(train_metrics_path, 'r') as f:
        tm = json.load(f)
    train_m = tm['train']
    test_m = tm['test']
    epochs = sorted(int(e) for e in train_m.keys() if e.isdigit())
    train_loss = [train_m[str(e)]['loss'] for e in epochs]
    test_loss = [test_m[str(e)]['loss'] for e in epochs]
    train_acc = [train_m[str(e)]['acc'] for e in epochs]
    test_acc = [test_m[str(e)]['acc'] for e in epochs]
    
    #ax_loss.plot(epochs, train_loss, label=f'{label} Train')
    #ax_loss.plot(epochs, test_loss, label=f'{label} Test', linestyle='--')
    ax_acc.plot(epochs, train_acc, label=f'{label} Train')
    ax_acc.plot(epochs, test_acc, label=f'{label} Test', linestyle='--')
    
    all_lrs, batch_idx, all_zetas = [], [], []
    for epoch, metrics in train_m.items():
        if isinstance(metrics, dict):
            zeta_list = metrics.get('zeta', [])
            lr_list = metrics.get('lr', [])
            nb = len(lr_list)
            e_idx = int(epoch)
            for b, lr in enumerate(lr_list):
                all_lrs.append(lr)
                batch_idx.append((e_idx-1)*nb + b)
            for zeta in zeta_list:
                if zeta is not None:
                    all_zetas.append(zeta)
    return (batch_idx, all_lrs), (batch_idx, all_zetas)

def plot_stepsize_schedule(lr_data, zeta_data, acc_values, acc_positions=None):
    fig_lr, (ax_sch, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))

    # CIFAR-10 constants
    TRAIN_SAMPLES = 50000
    BATCH_SIZE = 64
    EPOCHS = 200
    batches_per_epoch = TRAIN_SAMPLES // BATCH_SIZE

    for lbl, (batch_idx, lrs) in lr_data.items():
        epochs_x = [b_idx / batches_per_epoch for b_idx in batch_idx]
        ax_sch.plot(epochs_x, lrs, label=lbl, alpha=0.7)

        # Use density histogram
        counts, bins, _ = ax_hist.hist(lrs, bins=50, alpha=0.5, label=lbl, density=True)

        # Accuracy annotation
        if acc_positions and lbl in acc_positions:
            pos = acc_positions[lbl]
            x = bins[0] + pos['x'] * (bins[-1] - bins[0])
            y = pos['y'] * max(counts)
            ha = pos.get('ha', 'right')
            va = pos.get('va', 'top')
        else:
            x = bins[0] + 0.95 * (bins[-1] - bins[0])
            y = 0.9 * max(counts)
            ha = 'right'
            va = 'top'

        #ax_hist.text(
        #    x, y, f"{(acc_values[lbl]*100 + 1):.2f}", ha=ha, va=va,
        #    fontsize=14, bbox=dict(facecolor='white', alpha=0.8)
        #)

    # Format epoch-based x-axis
    ax_sch.set_xlabel('Epoch', fontsize=14)
    ax_sch.set_ylabel('Learning Rate', fontsize=14)
    #ax_sch.set_title('LR Schedule', fontsize=14)
    ax_sch.set_xlim(0, EPOCHS)
    ax_sch.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax_sch.xaxis.set_minor_locator(plt.MultipleLocator(5))

    # Set all tick label sizes to 14
    ax_sch.tick_params(axis='both', labelsize=14)
    ax_hist.tick_params(axis='both', labelsize=14)

    # Histogram plot formatting
    ax_hist.set_xlabel('Learning Rate', fontsize=14)
    ax_hist.set_ylabel('Density', fontsize=14)
    #ax_hist.set_title('LR Distribution', fontsize=14)
    ax_hist.legend(fontsize=14)
    ax_hist.grid(True)

    plt.tight_layout()

    fig_zeta,ax_zeta = plt.subplots(1,1,figsize=(14,5))
    for label,(bidx,zetas) in zeta_data.items():
        if len(zeta_data[label][1]) > 0:
            ax_zeta.plot(bidx,zetas,label=label)
    ax_zeta.set(xlabel='Batch idx',ylabel='Zeta',title='Zeta'); ax_zeta.legend(); ax_zeta.grid(True)

def process_eval_metrics(dir_path, label, ood_dataset_name, ensemble_size=16):
    with open(os.path.join(dir_path, f'eval_metrics_{ensemble_size}.json'), 'r') as f:
        em = json.load(f)
    acc = em['accuracy']
    nll =  em['nll']
    probs = np.array(em['probs'])
    y_true = np.array(em['y'])
    MI_vals = np.array(em['mutual_information'])
    pred = np.argmax(probs, axis=1)
    num_classes = len(np.unique(y_true))

    ood_info = None
    mi_compute = False
    if ood_dataset_name in em:
        oem = em[ood_dataset_name]
        H_ood = np.array(oem['entropy'])
        H_id = np.array(em['entropy'])
        all_entropy = np.concatenate([H_id, H_ood])
        labels_entropy = np.concatenate([np.zeros_like(H_id), np.ones_like(H_ood)])
        auroc_entropy = roc_auc_score(labels_entropy, all_entropy)
        acc_ood = oem['accuracy']
        nll_ood = oem['nll']
        ood_info = (auroc_entropy, acc_ood, nll_ood)
        mi_compute = True

    mi_mat = np.zeros((num_classes, num_classes))
    if mi_compute:
        counts = np.zeros((num_classes, num_classes))
        for t, p, mi in zip(y_true, pred, MI_vals):
            mi_mat[t, p] += mi
            counts[t, p] += 1
        mi_mat /= np.maximum(counts, 1)
    
    entropy_vals = np.array(em['entropy'])
    correct = (y_true == pred)
    prob_correct = probs[np.arange(len(y_true)), pred]
    calib_entry = (correct, prob_correct)
    
    
    return mi_mat, entropy_vals, calib_entry, ood_info, pred, y_true, acc, nll

def finalize_loss_acc_plots(ax_acc):
    for ax, ylabel, title in [#(ax_loss, 'Loss', 'Train vs Test Loss'),
                              (ax_acc, 'Accuracy', 'Train vs Test Accuracy')]:
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

def plot_mi_matrices(mi_matrices):
    n = len(mi_matrices)
    fig_mi, axes_mi = plt.subplots(1, n, figsize=(4*n,4))
    if n == 1:
        axes_mi = [axes_mi]
    for ax, (label, mat) in zip(axes_mi, mi_matrices.items()):
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=mat.max())
        ax.set_title(f'MI: {label}')
        ax.set_xlabel('Pred')
        ax.set_ylabel('True')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', fontsize=6)
    fig_mi.colorbar(im, ax=axes_mi, orientation='vertical', fraction=0.046, pad=0.04)

def plot_entropy_ecdfs(entropy_data, pred, y_true):
    fig_ecdf, ax_ecdf = plt.subplots(figsize=(8,6))
    for label, ent in entropy_data.items():
        mis_idx = (pred != y_true)
        ent_mis = ent[mis_idx]
        if ent_mis.size == 0:
            continue
        s = np.sort(ent_mis)
        ecdf = np.arange(1, len(s)+1) / len(s)
        ax_ecdf.plot(s, ecdf, label=label)
    ax_ecdf.set(xlabel='Entropy', ylabel='ECDF', title='Entropy vs ECDF over Misclassified points')
    ax_ecdf.legend()
    ax_ecdf.grid(True)
    
    fig_ecdf_cor, ax_ecdf_cor = plt.subplots(figsize=(8,6))
    for label, ent in entropy_data.items():
        cor_idx = (pred == y_true)
        ent_cor = ent[cor_idx]
        if ent_cor.size == 0:
            continue
        s = np.sort(ent_cor)
        ecdf = np.arange(1, len(s)+1) / len(s)
        ax_ecdf_cor.plot(s, ecdf, label=label)
    ax_ecdf_cor.set(xlabel='Entropy', ylabel='ECDF', title='Entropy vs ECDF over Correct points')
    ax_ecdf_cor.legend()
    ax_ecdf_cor.grid(True)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cifar.config as cf 
from models.resnet import ResNet18

@torch.no_grad()
def evaluate_accuracy(model, dataloader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def interpolate_weights(model, state_dict1, state_dict2, alpha):
    interpolated = {}
    for key in state_dict1:
        w1 = state_dict1[key]
        w2 = state_dict2[key]
        interpolated[key] = (1 - alpha) * w1 + alpha * w2
    model.load_state_dict(interpolated)


def compare_interpolation_accuracies(model_pairs, test_loader, device='cuda', steps=20):
    """
    Plots interpolation accuracy for multiple methods given a dict of checkpoint path pairs.
    """
    alphas = np.linspace(0, 1, steps)
    plt.figure(figsize=(7, 5))

    for label, (ckpt_path1, ckpt_path2) in model_pairs.items():
        model = ResNet18(num_classes=10).to(device).eval()
        sd1 = torch.load(ckpt_path1, map_location='cpu')
        sd2 = torch.load(ckpt_path2, map_location='cpu')
        state_dict1 = sd1.get('state_dict', sd1)
        state_dict2 = sd2.get('state_dict', sd2)

        accs = []
        for alpha in tqdm(alphas, desc=f"Interpolating [{label}]"):
            interpolate_weights(model, state_dict1, state_dict2, alpha)
            acc = evaluate_accuracy(model, test_loader, device=device)
            accs.append(acc)

        plt.plot(alphas, np.array(accs) * 100, marker='o', label=label)

    # Final plot formatting
    plt.title("Interpolation Accuracy on CIFAR-10", fontsize=14)
    plt.xlabel("Interpolation Coefficient (α)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    torch.cuda.empty_cache()
    gc.collect()

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_weight_projection(
    model_dirs,
    method="MDS",
    title="Weight Space Projection",
    random_state=0,
    n_components=2
):
    """
    Visualizes model weights using dimensionality reduction (MDS or PCA).

    Args:
        model_dirs (dict): Keys are labels (e.g., "SGD"), values are lists of directories with .pt/.pth checkpoints.
        method (str): "MDS" or "PCA"
        title (str): Plot title.
        random_state (int): Random seed for reproducibility.
        n_components (int): Number of dimensions to project down to (default: 2).
    """
    all_labels, all_coords = [], []

    for label, dir_list in model_dirs.items():
        for dir_path in dir_list:
            ckpt_files = sorted([
                f for f in os.listdir(dir_path)
                if f.endswith(('.pt', '.pth'))
            ])
            if not ckpt_files:
                print(f"Warning: No checkpoints in {dir_path}")
                continue

            weight_vecs = []
            for ckpt in ckpt_files:
                try:
                    state = torch.load(os.path.join(dir_path, ckpt), map_location='cpu')
                    sd = state.get('state_dict', state)
                    flat = np.concatenate([
                        p.detach().cpu().numpy().ravel()
                        for p in sd.values()
                        if isinstance(p, torch.Tensor)
                    ])
                    weight_vecs.append(flat)
                except Exception as e:
                    print(f"Failed to load {ckpt} in {dir_path}: {e}")
                    continue

            if len(weight_vecs) > 1:
                X = np.vstack(weight_vecs)
                X = StandardScaler().fit_transform(X)

                if method.upper() == "MDS":
                    reducer = MDS(n_components=n_components, random_state=random_state, n_init=4, max_iter=500)
                elif method.upper() == "PCA":
                    reducer = PCA(n_components=n_components, random_state=random_state)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                coords = reducer.fit_transform(X)
                all_coords.append(coords)
                all_labels.append(label)

    # Plotting
    plt.figure(figsize=(6, 6))
    for label, coords in zip(all_labels, all_coords):
        plt.scatter(coords[:, 0], coords[:, 1], label=label, alpha=0.7)

    plt.title(title, fontsize=14)
    plt.xlabel("Component 1", fontsize=14)
    plt.ylabel("Component 2", fontsize=14)
    plt.legend(fontsize=12)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_metrics(model_runs, ood_dataset_name='stl10', ensemble_size=16):
    """
    Evaluate metrics across multiple runs, compute averages and standard deviations.
    
    Args:
        model_runs (dict): Mapping from model labels to lists of run directories.
        ood_dataset_name (str): Name of the OOD dataset to evaluate (default: 'stl10').
    """
    metrics_summary = {}
    
    for label, run_dirs in model_runs.items():
        acc_list, nll_list, ece_list = [], [], []
        auroc_list, acc_ood_list, nll_ood_list = [], [], []
        
        for run_dir in run_dirs:
            # Load evaluation metrics
            if 'csgld' in run_dir:
                eval_path = os.path.join(run_dir, 'eval_metrics_12.json')
            else:
                eval_path = os.path.join(run_dir, f'eval_metrics_{ensemble_size}.json')
            try:
                with open(eval_path, 'r') as f:
                    em = json.load(f)
            except FileNotFoundError:
                print(f"Warning: {eval_path} not found. Skipping.")
                continue
            
            # Basic metrics
            acc = em['accuracy']
            nll = em['nll']
            
            # ECE calculation
            probs = np.array(em['probs'])
            y_true = np.array(em['y'])
            pred = np.argmax(probs, axis=1)
            correct = (y_true == pred)
            prob_correct = probs[np.arange(len(y_true)), pred]
            ece = compute_ece(correct.astype(int), prob_correct)
            
            acc_list.append(acc)
            nll_list.append(nll)
            ece_list.append(ece)
            
            # OOD metrics
            if ood_dataset_name in em or 'ood' in em:
                if ood_dataset_name in em:
                    oem = em[ood_dataset_name]
                if 'ood' in em:
                    oem = em['ood']
                H_ood = np.array(oem['entropy'])
                H_id = np.array(em['entropy'])
                all_entropy = np.concatenate([H_id, H_ood])
                labels_entropy = np.concatenate([np.zeros_like(H_id), np.ones_like(H_ood)])
                auroc = roc_auc_score(labels_entropy, all_entropy)
                acc_ood = oem['accuracy']
                nll_ood = oem['nll']
                
                auroc_list.append(auroc)
                acc_ood_list.append(acc_ood)
                nll_ood_list.append(nll_ood)
        
        # Compute averages and standard deviations
        metrics = {
            'acc': (np.mean(acc_list)), 'acc_std': np.std(acc_list),
            'nll': (np.mean(nll_list)), 'nll_std': np.std(nll_list),
            'ece': (np.mean(ece_list)), 'ece_std': np.std(ece_list),
        }
        if auroc_list:
            metrics.update({
                'auroc': np.mean(auroc_list),
                'auroc_std': np.std(auroc_list),
                'acc_ood': np.mean(acc_ood_list),
                'acc_ood_std': np.std(acc_ood_list),
                'nll_ood': np.mean(nll_ood_list),
                'nll_ood_std': np.std(nll_ood_list),
            })
        
        metrics_summary[label] = metrics
    
    # Print results
    for label, metrics in metrics_summary.items():
        print(f"\n=== {label} ===")
        print(f"CIFAR-10 Accuracy: {metrics['acc']:.4f} ± {metrics['acc_std']:.4f}")
        print(f"CIFAR-10 NLL:      {metrics['nll']:.4f} ± {metrics['nll_std']:.4f}")
        print(f"CIFAR-10 ECE:      {100 * metrics['ece']:.4f} ± {100 * metrics['ece_std']:.4f}")
        if 'auroc' in metrics:
            print(f"\n{ood_dataset_name}:")
            print(f"  AUROC:           {metrics['auroc']:.3f} ± {metrics['auroc_std']:.3f}")
            print(f"  Accuracy:        {metrics['acc_ood']:.4f} ± {metrics['acc_ood_std']:.4f}")
            print(f"  NLL:             {metrics['nll_ood']:.4f} ± {metrics['nll_ood_std']:.4f}")

    
def plot_lr_schedule():
    zeta = np.linspace(0, 0.000025, 1000)

    # First set of parameters
    m1, M1, r1 = 0.1, 10, 0.15
    f1 = 0.2 * m1 * (zeta ** r1 + M1) / (zeta ** r1 + m1)

    # Second set of parameters
    m2, M2, r2 = 0.02, 2, 0.25
    f2 = m2 * (zeta ** r2 + M2) / (zeta ** r2 + m2)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.semilogy(zeta, f1, label=f'm=0.1, M=10, r={r1}', linewidth=2)
    plt.semilogy(zeta, f2, label=f'm=0.02, M=2, r={r2}', linewidth=2, linestyle='--')
    plt.xlabel(r'$\zeta$', fontsize=14)
    plt.ylabel(r'$f(\zeta) = m \cdot \frac{\zeta^r + M}{\zeta^r + m}$', fontsize=14)
    plt.title('Learning Rate Scaling Function', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

from scipy.stats import sem  # Standard error of the mean

def print_stats(label, values):
    mean = np.mean(values)
    error = 3 * np.std(values)
    lower = mean - error
    upper = mean + error
    print(f"{label}: Mean = {mean:.4f}, 95% CI = [{lower:.4f}, {upper:.4f}]")

    # Print LR stats
# fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
# print("-----------CIFAR-10---------")
# dir_path = "cifar/ckpt_sgld/sa-sgld/S1/lr0.02_a5.0_m0.1_M10.0_r0.25"
# label = "Alpha=5, r = 0.25"
# lr_entry, zeta_entry = process_train_metrics(dir_path, label, ax_acc)
# print_stats(label, lr_entry[1])

import matplotlib.pyplot as plt
import seaborn as sns
import ast

def plot_stepsizes_and_zetas(df, M, alpha, dtau, omega, r, max_runs=None):
    """
    Plot step sizes and zetas traces and distributions for given filters.
    Optimized version for faster execution.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    M, alpha, dtau, omega, r (scalar or list): Filter values for the parameters.
    max_runs (int or None): Max number of rows to plot for speed. None = all.
    """
    
    # Convert scalars to lists
    if not isinstance(M, list): M = [M]
    if not isinstance(alpha, list): alpha = [alpha]
    if not isinstance(dtau, list): dtau = [dtau]
    if not isinstance(omega, list): omega = [omega]
    if not isinstance(r, list): r = [r]
    
    # Filter matching rows
    matching_rows = df[
        df["M"].isin(M) &
        df["alpha"].isin(alpha) &
        df["dtau"].isin(dtau) &
        df["Omega"].isin(omega) &
        df["r"].isin(r)
    ].copy()
    
    if matching_rows.empty:
        print("No matching rows found.")
        return
    
    if max_runs is not None and len(matching_rows) > max_runs:
        print(f"Limiting plot to first {max_runs} runs out of {len(matching_rows)} matched.")
        matching_rows = matching_rows.iloc[:max_runs]
    
    print("\nMatched rows:\n", matching_rows[["M", "alpha", "dtau", "Omega", "r", "Acc", "NLL", "ECE", "OOD_AUC"]])
    
    # Optimized parsing function
    def fast_parse(series):
        """Parse string representations to lists, with error handling"""
        def safe_eval(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            except (ValueError, SyntaxError):
                return []
        return series.apply(safe_eval)
    
    # Parse all needed columns at once
    print("Parsing data...")
    step_sizes_data = fast_parse(matching_rows["_step_sizes"])
    batch_idx_data = fast_parse(matching_rows["_batch_idx"])
    zetas_data = fast_parse(matching_rows["_zetas"])
    zeta_batch_idx_data = fast_parse(matching_rows["_zeta_batch_idx"])
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    
    # Process each run with optimizations
    for idx, (i, row) in enumerate(matching_rows.iterrows()):
        label = f"Run {idx+1}: M={row['M']}, α={row['alpha']}, dtau={row['dtau']}, Ω={row['Omega']}, r={row['r']}"
        
        # Get data as numpy arrays for faster operations
        try:
            steps = np.array(step_sizes_data.iloc[idx], dtype=float)
            bidx = np.array(batch_idx_data.iloc[idx], dtype=float)
            zetas = np.array(zetas_data.iloc[idx], dtype=float)
            zeta_bidx = np.array(zeta_batch_idx_data.iloc[idx], dtype=float)
        except (ValueError, TypeError):
            print(f"Skipping run {idx+1} due to data parsing issues")
            continue
        
        # Skip empty or invalid data
        if len(steps) == 0 or len(zetas) == 0:
            print(f"Skipping run {idx+1} due to empty data")
            continue
        
        # Aggressive subsampling for large datasets to speed up plotting
        max_trace_points = 5000  # Reduced from potential millions
        max_kde_points = 2000    # For distribution plots
        
        # Subsample step size data
        if len(steps) > max_trace_points:
            trace_idx = np.linspace(0, len(steps)-1, max_trace_points, dtype=int)
            steps_trace = steps[trace_idx]
            bidx_trace = bidx[trace_idx]
        else:
            steps_trace = steps
            bidx_trace = bidx
            
        if len(steps) > max_kde_points:
            kde_idx = np.random.choice(len(steps), max_kde_points, replace=False)
            steps_kde = steps[kde_idx]
        else:
            steps_kde = steps
        
        # Subsample zeta data
        if len(zetas) > max_trace_points:
            trace_idx = np.linspace(0, len(zetas)-1, max_trace_points, dtype=int)
            zetas_trace = zetas[trace_idx]
            zeta_bidx_trace = zeta_bidx[trace_idx]
        else:
            zetas_trace = zetas
            zeta_bidx_trace = zeta_bidx
            
        if len(zetas) > max_kde_points:
            kde_idx = np.random.choice(len(zetas), max_kde_points, replace=False)
            zetas_kde = zetas[kde_idx]
        else:
            zetas_kde = zetas
        
        # Plot with reduced line width for performance
        color = plt.cm.tab10(idx % 10)  # Cycle through colors
        alpha_val = 0.7 if len(matching_rows) > 5 else 1.0
        
        # Traces (use thinner lines for better performance with many runs)
        lw = 0.8 if len(matching_rows) > 10 else 1.0
        axs[0, 0].plot(bidx_trace, steps_trace, lw=lw, label=label, color=color, alpha=alpha_val)
        axs[1, 0].plot(zeta_bidx_trace, zetas_trace, lw=lw, label=label, color=color, alpha=alpha_val)
        
        # Distributions with subsampled data
        try:
            sns.kdeplot(steps_kde, ax=axs[0, 1], label=label, linewidth=1.2, 
                       color=color, alpha=alpha_val)
            sns.kdeplot(zetas_kde, ax=axs[1, 1], label=label, linewidth=1.2, 
                       color=color, alpha=alpha_val)
        except Exception as e:
            print(f"KDE plot failed for run {idx+1}: {e}")
    
    # Set titles and labels
    axs[0, 0].set_title("Step Size Trace")
    axs[0, 0].set_xlabel("Batch Index")
    axs[0, 0].set_ylabel("Step Size")
    
    axs[0, 1].set_title("Step Size Distribution")
    axs[0, 1].set_xlabel("Step Size")
    axs[0, 1].set_ylabel("Density")
    
    axs[1, 0].set_title("Zeta Trace")
    axs[1, 0].set_xlabel("Batch Index")
    axs[1, 0].set_ylabel("Zeta")
    
    axs[1, 1].set_title("Zeta Distribution")
    axs[1, 1].set_xlabel("Zeta")
    axs[1, 1].set_ylabel("Density")
    
    # Smart legend handling - avoid legends for too many runs as they slow down rendering
    if len(matching_rows) <= 8:
        for ax in axs.flat:
            ax.legend(fontsize=8 if len(matching_rows) > 4 else 10)
    else:
        print(f"Legend omitted for performance (plotting {len(matching_rows)} runs)")
    
    plt.tight_layout()
    plt.show()




import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import re
from tqdm import tqdm
import concurrent.futures

def extract_params_from_dirname(dirname):
    pattern = r"lr([0-9.]+)_a([0-9.]+)_m([0-9.]+)_M([0-9.]+)_r([0-9.]+)(?:_O([0-9.]+))?"
    match = re.match(pattern, dirname)
    if not match:
        return None, None, None, None
    dtau = float(match.group(1))
    alpha = float(match.group(2))
    m = float(match.group(3))
    M = float(match.group(4))
    r = float(match.group(5))
    omega = float(match.group(6)) if match.group(6) else 1.0
    return alpha, dtau, omega, r, m, M

def extract_metrics_from_dir(dir_path):
    eval_path = os.path.join(dir_path, "eval_metrics_16.json")
    train_path = os.path.join(dir_path, "train_metrics.json")

    if not os.path.exists(eval_path) or not os.path.exists(train_path):
        return None

    with open(eval_path, 'r') as f:
        em = json.load(f)

    acc = em.get("accuracy")
    nll = em.get("nll")
    y_true = np.array(em["y"])
    probs = np.array(em["probs"])
    pred = np.argmax(probs, axis=1)
    correct = (y_true == pred)
    prob_correct = probs[np.arange(len(y_true)), pred]
    ece = compute_ece(correct.astype(int), prob_correct)

    if 'ood' in em:
        oem = em['ood']
    elif 'stl10' in em:
        oem = em['stl10']

    try:
        H_ood = np.array(oem["entropy"])
        H_id = np.array(em["entropy"])
        all_entropy = np.concatenate([H_id, H_ood])
        labels_entropy = np.concatenate([np.zeros_like(H_id), np.ones_like(H_ood)])
        auroc_entropy = roc_auc_score(labels_entropy, all_entropy)
    except:
        auroc_entropy = np.nan

    with open(train_path, 'r') as f:
        tm = json.load(f)["train"]

    # Extract learning rates and batch indices
    all_stepsizes, bidx = [], []
    # Extract zetas and their batch indices
    all_zetas, zeta_bidx = [], []

    for epoch, metrics in tm.items():
        if isinstance(metrics, dict):
            lr_list = metrics.get('lr', [])
            zeta_list = metrics.get('zeta', [])
            nb = len(lr_list)
            e_idx = int(epoch)

            for b, lr in enumerate(lr_list):
                all_stepsizes.append(lr)
                bidx.append((e_idx - 1) * nb + b)

            for b, zeta in enumerate(zeta_list):
                all_zetas.append(zeta)
                zeta_bidx.append((e_idx - 1) * len(zeta_list) + b)

    if all_stepsizes:
        all_stepsizes = all_stepsizes[1:]
        bidx = bidx[1:]
        all_stepsizes_np = np.array(all_stepsizes)
        min_lr = np.min(all_stepsizes_np)
        mean_lr = np.mean(all_stepsizes_np)
        max_lr = np.max(all_stepsizes_np)
        std_lr = np.std(all_stepsizes_np)
    else:
        min_lr = mean_lr = max_lr = std_lr = np.nan

    if all_zetas:
        all_zetas = all_zetas[1:]
        zeta_bidx = zeta_bidx[1:]
        all_zetas_np = np.array(all_zetas)
        min_zeta = np.min(all_zetas_np)
        mean_zeta = np.mean(all_zetas_np)
        max_zeta = np.max(all_zetas_np)
        std_zeta = np.std(all_zetas_np)
    else:
        min_zeta = mean_zeta = max_zeta = std_zeta = np.nan

    return (acc, nll, ece, auroc_entropy,
            (min_lr, mean_lr, max_lr, std_lr), all_stepsizes, bidx,
            (min_zeta, mean_zeta, max_zeta, std_zeta), all_zetas, zeta_bidx)
