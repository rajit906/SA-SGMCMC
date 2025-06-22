import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
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

def cifar_10c(ensemble, corrupt_files, labels_c10c, normalize_c10, c10c_root, batch_size, device):
    # metrics arrays
    e_acc = np.zeros((5, len(corrupt_files)))
    e_nll = np.zeros((5, len(corrupt_files)))
    # prepare containers for all probs & labels
    all_probs  = []   # 5 severities × N corruptions
    all_labels = []
    for sev in range(5):  # severity levels 0..4 correspond to 1..5
        sev_probs  = []
        sev_labels = []
        for idx, cfname in enumerate(corrupt_files):
            arr = np.load(os.path.join(c10c_root, cfname))
            arr = arr.reshape(5, 10000, 32, 32, 3)
            imgs = arr[sev]
            imgs = torch.from_numpy(imgs.transpose(0,3,1,2)).float().div(255.0)
            imgs = torch.stack([normalize_c10(img) for img in imgs])
            labels_sev = labels_c10c[sev]
            # build DataLoader
            ds     = TensorDataset(imgs, torch.from_numpy(labels_sev))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            # get predictions
            probs_c, y_c, _, _, _ = predict(loader, ensemble, device)
            preds_c = np.argmax(probs_c, axis=1)
            # record metrics
            e_acc[sev, idx] = accuracy_score(y_c, preds_c)
            e_nll[sev, idx] = log_loss(y_c, probs_c)
            # stash for indexing later
            sev_probs.append(probs_c)
            sev_labels.append(y_c)
        # after this severity
        mean_acc = e_acc[sev].mean()
        mean_nll = e_nll[sev].mean()
        print(f"CIFAR-10C Severity {sev+1}: "
            f"Mean Acc={mean_acc:.4f}, Mean NLL={mean_nll:.4f}")
        all_probs.append(sev_probs.tolist())
        all_labels.append(sev_labels.tolist())
    return e_acc, e_nll, all_probs, all_labels

# # ---------- CIFAR 10-C -------------
#     if args.test_corruption:
#         c10c_dir = os.path.join(args.data_path, 'CIFAR-10-C')
#         if not os.path.isdir(c10c_dir):
#             import urllib.request, tarfile
#             url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1'
#             fname = os.path.join(args.data_path, 'CIFAR-10-C.tar')
#             print(f"Downloading CIFAR-10-C to {fname}...")
#             urllib.request.urlretrieve(url, fname)
#             print("Extracting CIFAR-10-C...")
#             with tarfile.open(fname) as tar:
#                 tar.extractall(path=args.data_path)
#             os.remove(fname)

#         c10c_root   = os.path.join(args.data_path, 'CIFAR-10-C')
#         labels_raw  = np.load(os.path.join(c10c_root, 'labels.npy'))
#         if labels_raw.shape[0] == 50000:
#             labels_c10c = labels_raw.reshape(5, 10000)
#         elif labels_raw.shape[0] == 10000:
#             labels_c10c = np.tile(labels_raw, (5,1))
#         else:
#             raise ValueError(f"Unexpected labels.npy size: {labels_raw.shape}")

#         normalize_c10 = transforms.Normalize((0.4914,0.4822,0.4465),
#                                             (0.2023,0.1994,0.2010))
#         corrupt_files = sorted([f for f in os.listdir(c10c_root)
#                                 if f.endswith('.npy') and f!='labels.npy'])

#         e_acc, e_nll, all_probs, all_labels = cifar_10c(ensemble, corrupt_files, labels_c10c, 
#                                                               normalize_c10, c10c_root, args.batch_size, device)

#         # attach to output dict
#         out['cifar-10c'] = {
#             'accuracy': e_acc.tolist(), # shape (5, n_corruptions)
#             'nll'     : e_nll.tolist(), # same shape
#             'probs'   : all_probs,      # list-of-lists: [severity][corruption] -> array
#             'y'       : all_labels      # true labels with same indexing
#         }
# # --------- End of CIFAR-10C ----------------

# # === Compute MDS of weight samples ===
    # all_labels, coords_list = [], []
    # for label, dir_path in model_dirs.items():
    #     # Collect all checkpoint files (.pt or .pth)
    #     ckpts = sorted([f for f in os.listdir(dir_path) if f.endswith(('.pt', '.pth'))])
    #     w_vecs = []
    #     for ckpt in ckpts:
    #         state = torch.load(os.path.join(dir_path, ckpt), map_location='cpu')
    #         # handle nested dicts
    #         sd = state.get('state_dict', state)
    #         flat = np.concatenate([p.detach().cpu().numpy().ravel() for p in sd.values()])
    #         w_vecs.append(flat)
    #     if len(w_vecs) > 1:
    #         coords = MDS(n_components=2, random_state=0).fit_transform(np.vstack(w_vecs))
    #         coords_list.append(coords)
    #         all_labels.append(label)
    # # Plot ensemble MDS
    # plt.figure(figsize=(6,6))
    # for label, coords in zip(all_labels, coords_list):
    #     plt.scatter(coords[:,0], coords[:,1], label=label)
    # plt.title("MDS of Weight Samples")
    # plt.xlabel("Component 1")
    # plt.ylabel("Component 2")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    #print(f"{label} STL-10: Acc={oem['accuracy']:.4f}, NLL={oem['nll']:.4f}")
            # H_ood = oem['entropy']
            # H_id = em['entropy']
            # fig_io, ax_io = plt.subplots(figsize=(8,6))
            # ax_io.hist(H_id,  bins=50, alpha=0.5, density=True,
            #        label=f'ID {label}')
            # ax_io.hist(H_ood, bins=50, alpha=0.5, density=True,
            #        label=f'OOD {label}')
            # ax_io.set_xlabel('Predictive Entropy')
            # ax_io.set_ylabel('Density')
            # ax_io.set_title('In- vs. Out-of-Distribution Entropy Histograms')
            # ax_io.legend()
            # ax_io.grid(True)