import os
import json
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from utils import compute_ece

def extract_number_from_path(dir_path):
    """Extract the number parameter from path like ./ckpt_sgld/sgld/0.2/S1"""
    parts = dir_path.split('/')
    # Find the number between 'sgld' and seed directory (e.g., 'S1')
    try:
        sgld_idx = parts.index('sgld')
        if sgld_idx + 1 < len(parts):
            number_str = parts[sgld_idx + 1]
            return float(number_str)
    except (ValueError, IndexError):
        pass
    return None

def extract_seed_from_path(dir_path):
    """Extract seed identifier from path like ./ckpt_sgld/sgld/0.2/S1"""
    parts = dir_path.split('/')
    if parts and parts[-1].startswith('S'):
        return parts[-1]
    return None

def extract_metrics_from_dir(dir_path):
    """Extract metrics from evaluation and training files"""
    eval_path = os.path.join(dir_path, "eval_metrics_16.json")
    train_path = os.path.join(dir_path, "train_metrics.json")
    
    if not os.path.exists(eval_path) or not os.path.exists(train_path):
        return None
    
    # Load evaluation metrics
    with open(eval_path, 'r') as f:
        em = json.load(f)
    
    acc = em.get("accuracy")
    nll = em.get("nll")
    
    # Calculate ECE
    y_true = np.array(em["y"])
    probs = np.array(em["probs"])
    pred = np.argmax(probs, axis=1)
    correct = (y_true == pred)
    prob_correct = probs[np.arange(len(y_true)), pred]
    ece = compute_ece(correct.astype(int), prob_correct)
    
    # Calculate OOD AUROC
    try:
        if 'ood' in em:
            oem = em['ood']
        elif 'stl10' in em:
            oem = em['stl10']
        else:
            auroc_entropy = np.nan
            return (acc, nll, ece, auroc_entropy)
        
        H_ood = np.array(oem["entropy"])
        H_id = np.array(em["entropy"])
        all_entropy = np.concatenate([H_id, H_ood])
        labels_entropy = np.concatenate([np.zeros_like(H_id), np.ones_like(H_ood)])
        auroc_entropy = roc_auc_score(labels_entropy, all_entropy)
    except:
        auroc_entropy = np.nan
    
    return (acc, nll, ece, auroc_entropy)

def process_directory(dir_path):
    """Process a single directory and extract all metrics"""
    if not os.path.isdir(dir_path):
        return None
    
    # Extract number parameter and seed from path
    number = extract_number_from_path(dir_path)
    seed = extract_seed_from_path(dir_path)
    
    if number is None or seed is None:
        return None
    
    # Extract metrics
    metrics = extract_metrics_from_dir(dir_path)
    if metrics is None:
        return None
    
    acc, nll, ece, auroc = metrics
    
    return {
        "number": number,
        "seed": seed,
        "Acc": acc * 100,
        "NLL": nll,
        "ECE": ece * 100,
        "OOD_AUC": auroc * 100,
    }

def find_sgld_directories(base_path):
    """Find all SGLD directories matching the pattern ./ckpt_sgld/sgld/**/S*"""
    sgld_dirs = []
    sgld_base = os.path.join(base_path, "ckpt_sgld", "sgld")
    
    if not os.path.exists(sgld_base):
        print(f"Base directory {sgld_base} does not exist")
        return sgld_dirs
    
    # Look for directories with pattern number/S*
    for item in os.listdir(sgld_base):
        item_path = os.path.join(sgld_base, item)
        if os.path.isdir(item_path):
            # Look for all seed directories (S1, S2, S3, etc.)
            for seed_item in os.listdir(item_path):
                if seed_item.startswith('S'):
                    seed_path = os.path.join(item_path, seed_item)
                    if os.path.exists(seed_path) and os.path.isdir(seed_path):
                        sgld_dirs.append(seed_path)
    
    return sgld_dirs

def format_mean_std(mean_val, std_val, multiply_by_100=False):
    """Format mean ± std with appropriate precision"""
    if multiply_by_100:
        mean_val *= 100
        std_val *= 100
    
    if np.isnan(mean_val) or np.isnan(std_val):
        return "NaN ± NaN"
    
    # Determine precision based on the magnitude of std
    if std_val < 0.01:
        precision = 3
    elif std_val < 0.1:
        precision = 2
    else:
        precision = 1
    
    return f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"

def aggregate_results_by_stepsize(results):
    """Group results by stepsize and calculate mean ± std"""
    # Group by stepsize
    grouped = defaultdict(list)
    for result in results:
        grouped[result['number']].append(result)
    
    aggregated = []
    for stepsize, group_results in grouped.items():
        if len(group_results) == 0:
            continue
            
        # Convert to arrays for easier calculation
        acc_values = [r['Acc'] for r in group_results]
        nll_values = [r['NLL'] for r in group_results]
        ece_values = [r['ECE'] for r in group_results]
        ood_auc_values = [r['OOD_AUC'] for r in group_results if not np.isnan(r['OOD_AUC'])]
        
        # Calculate statistics
        acc_mean, acc_std = np.mean(acc_values), np.std(acc_values, ddof=1) if len(acc_values) > 1 else 0
        nll_mean, nll_std = np.mean(nll_values), np.std(nll_values, ddof=1) if len(nll_values) > 1 else 0
        ece_mean, ece_std = np.mean(ece_values), np.std(ece_values, ddof=1) if len(ece_values) > 1 else 0
        
        if len(ood_auc_values) > 0:
            ood_auc_mean = np.mean(ood_auc_values)
            ood_auc_std = np.std(ood_auc_values, ddof=1) if len(ood_auc_values) > 1 else 0
        else:
            ood_auc_mean, ood_auc_std = np.nan, np.nan
        
        aggregated.append({
            "stepsize": stepsize,
            "n_seeds": len(group_results),
            "Acc": format_mean_std(acc_mean, acc_std),
            "NLL": format_mean_std(nll_mean, nll_std),
            "ECE": format_mean_std(ece_mean, ece_std),
            "OOD_AUC": format_mean_std(ood_auc_mean, ood_auc_std),
            # Also keep raw values for potential further analysis
            "Acc_raw_mean": acc_mean,
            "Acc_raw_std": acc_std,
            "NLL_raw_mean": nll_mean,
            "NLL_raw_std": nll_std,
            "ECE_raw_mean": ece_mean,
            "ECE_raw_std": ece_std,
            "OOD_AUC_raw_mean": ood_auc_mean,
            "OOD_AUC_raw_std": ood_auc_std,
        })
    
    return aggregated

# --- Main script ---
base_dir = "./cifar"  # Adjust this to your base directory
sghmc_directories = find_sgld_directories(base_dir)

if not sghmc_directories:
    print("No SGHMC directories found!")
    exit(1)

print(f"Found {len(sghmc_directories)} SGLD directories")

# Process all directories
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_directory, d) for d in sghmc_directories]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        res = f.result()
        if res is not None:
            results.append(res)

if not results:
    print("No results found!")
    exit(1)

print(f"Successfully processed {len(results)} seed runs")

# Create individual results DataFrame
df_individual = pd.DataFrame(results)
df_individual = df_individual.sort_values(['number', 'seed'])

# Aggregate results by stepsize
aggregated_results = aggregate_results_by_stepsize(results)
df_aggregated = pd.DataFrame(aggregated_results)
df_aggregated = df_aggregated.sort_values('stepsize')

# Save both individual and aggregated results
os.makedirs("./tables", exist_ok=True)
df_individual.to_csv("./tables/sgld_cifar10_individual.csv", index=False)
df_aggregated.to_csv("./tables/sgld_cifar10_aggregated.csv", index=False)

print("\nAggregated Results (Mean ± Std):")
print(df_aggregated[['stepsize', 'n_seeds', 'Acc', 'NLL', 'ECE', 'OOD_AUC']].to_string(index=False))

print(f"\nSaved individual results to: ./tables/sgld_cifar10_individual.csv")
print(f"Saved aggregated results to: ./tables/sgld_cifar10_aggregated.csv")