import os
import json
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils import compute_ece
import os
from utils import plot_stepsizes_and_zetas, extract_params_from_dirname, extract_metrics_from_dir
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import os

def process_directory(dirname):
    dir_path = os.path.join(base_dir, dirname)
    if not os.path.isdir(dir_path):
        return None
    alpha, dtau, omega, r, m, M = extract_params_from_dirname(dirname)
    if alpha is None:
        return None

    metrics = extract_metrics_from_dir(dir_path)
    if metrics is None:
        return None

    (acc, nll, ece, auroc,
     (min_lr, mean_lr, max_lr, std_lr), stepsizes, bidx,
     (min_zeta, mean_zeta, max_zeta, std_zeta), zetas, zeta_bidx) = metrics

    return {
        "alpha": alpha,
        "dtau": dtau,
        "Omega": omega,
        "m": m,
        "M": M,
        "r": r,
        "Acc": acc * 100,
        "NLL": nll,
        "ECE": ece * 100,
        "OOD_AUC": auroc * 100,
        "Min Mean Max Std Stepsize": f"{min_lr:.4f} {mean_lr:.4f} {max_lr:.4f} {std_lr:.4f}",
        "Min Mean Max Std Zeta": f"{min_zeta:.4f} {mean_zeta:.4f} {max_zeta:.4f} {std_zeta:.4f}",
        "_step_sizes": str(list(stepsizes)),
        "_batch_idx": str(list(bidx)),
        "_zetas": str(list(zetas)),
        "_zeta_batch_idx": str(list(zeta_bidx)),
    }

# --- Main script ---

base_dir = "cifar/ckpt_sgld/sa-sgld_new/S1"
all_dirs = os.listdir(base_dir)

results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_directory, d) for d in all_dirs]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        res = f.result()
        if res is not None:
            results.append(res)

df = pd.DataFrame(results)
df.to_csv("./tables/sa-sgld_new_cifar10.csv", index=False)