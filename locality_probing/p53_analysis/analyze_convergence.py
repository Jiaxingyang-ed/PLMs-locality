#!/usr/bin/env python3
"""
Convergence Dynamics Analysis of Three Metrics (L_θ, L_m, L_r)

This script analyzes the convergence dynamics of three locality metrics for p53 mutations:
- L_θ: Angular mismatch between local and global embedding predictions
- L_m: Magnitude deviation (log ratio of norms)
- L_r: Residual error (normalized Euclidean distance)

The script fits exponential decay curves to each metric as a function of window radius
to extract the irreducible residual plateau 'c', which correlates with conformational heterogeneity.

Dependencies:
- torch, pickle, numpy, pandas
- scipy.optimize.curve_fit (for exponential fitting)
- matplotlib (for visualization)

Inputs:
- P53 mutation data with precomputed global embedding deltas (p53_mutation_deltas.pkl)
- Trained local encoder model (local_encoder_cosine.pt)

Outputs:
- Three-panel figure showing convergence of L_θ, L_m, L_r across window radii
- Fitted exponential parameters (A, α, c) for each metric and mutation
- Console output of fitted parameters

Scientific Purpose:
To quantify how prediction error converges as local window size increases, revealing
irreducible global dependencies that cannot be explained by local context alone.
"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from torch import nn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models import LocalSeqEncoder
from config import (DEVICE, MAX_ANALYSIS_WINDOW, LOCAL_ENCODER_COSINE,
                   P53_MUTATIONS_DIR, ANALYSIS_RADII, AA_TO_IDX)

MAX_WINDOW_LEN = MAX_ANALYSIS_WINDOW
MODEL_PATH = LOCAL_ENCODER_COSINE
DATA_PATH = os.path.join(P53_MUTATIONS_DIR, "p53_mutation_deltas.pkl")

model = LocalSeqEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
aa_to_idx = AA_TO_IDX

def compute_all_L(dh_local, dh_global):
    """
    Compute three locality metrics comparing local vs global embedding predictions.
    
    Args:
        dh_local: Local embedding prediction from sequence window
        dh_global: Global embedding delta from full sequence
    
    Returns:
        tuple: (L_theta, L_m, L_r) - angular, magnitude, and residual errors
    """
    norm_l = np.linalg.norm(dh_local)
    norm_g = np.linalg.norm(dh_global)
    eps = 1e-12  # Small constant to avoid division by zero
    
    if norm_g < eps:
        return 0.0, 0.0, 0.0
    
    # L_theta: Angular mismatch (1 - cosine similarity)
    # Measures directional difference between local and global predictions
    if norm_l < eps:
        L_theta = 1.0
    else:
        cos = np.dot(dh_local, dh_global) / (norm_l * norm_g)
        cos = np.clip(cos, -1.0, 1.0)
        L_theta = 1.0 - cos  # L_theta = 1 - cos(delta_local, delta_global)
    
    # L_m: Magnitude deviation (log ratio of norms)
    # Measures scale difference between local and global predictions
    L_m = np.abs(np.log((norm_l + eps) / (norm_g + eps)))
    
    # L_r: Residual error (normalized Euclidean distance)
    # Measures overall prediction error normalized by global magnitude
    L_r = np.linalg.norm(dh_local - dh_global) / (norm_g + eps)
    
    return L_theta, L_m, L_r

with open(DATA_PATH, 'rb') as f:
    p53_data = pickle.load(f)

radii = ANALYSIS_RADII
measures = {name: {'radii': radii, 'L_theta': [], 'L_m': [], 'L_r': [], 'category': s['category']}
            for name, s in zip([d['name'] for d in p53_data], p53_data)}

for sample in p53_data:
    name = sample["name"]
    local_seq = sample["local_seq_full"]
    center = sample["center_in_local"]
    dh_global = sample["delta_h_global"]
    for r in radii:
        start = max(0, center - r)
        end = min(len(local_seq), center + r + 1)
        sub_seq = local_seq[start:end]
        seq_idx = [aa_to_idx.get(aa, 0) for aa in sub_seq]
        if len(seq_idx) < MAX_WINDOW_LEN:
            seq_idx += [0] * (MAX_WINDOW_LEN - len(seq_idx))
        else:
            seq_idx = seq_idx[:MAX_WINDOW_LEN]
        seq_tensor = torch.tensor(seq_idx).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dh_local = model(seq_tensor, None).squeeze().cpu().numpy()
        L_theta, L_m, L_r = compute_all_L(dh_local, dh_global)
        measures[name]['L_theta'].append(L_theta)
        measures[name]['L_m'].append(L_m)
        measures[name]['L_r'].append(L_r)

# Fit exponential decay to each metric and plot
# Model: L(r) = A * exp(-alpha * r) + c
# where A = initial amplitude, alpha = decay rate, c = irreducible residual plateau
def exp_decay(r, A, alpha, c):
    """
    Exponential decay function for fitting convergence curves.
    
    Args:
        r: Window radius
        A: Initial amplitude (error at r=0)
        alpha: Decay rate (how quickly error decreases)
        c: Irreducible residual plateau (error that cannot be eliminated)
    
    Returns:
        Predicted error at radius r
    """
    return A * np.exp(-alpha * r) + c

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, title in zip(axes, ['L_theta', 'L_m', 'L_r'], ['Angular Mismatch', 'Magnitude Deviation', 'Residual Error']):
    for name, data in measures.items():
        r = np.array(data['radii'])
        y = np.array(data[metric])
        ax.scatter(r, y, label=name)
        try:
            # Fit exponential decay: L(r) = A * exp(-alpha * r) + c
            popt, _ = curve_fit(exp_decay, r, y, p0=[5, 0.1, 3], maxfev=10000)
            # Extract fitted parameters
            amplitude_A = popt[0]  # Initial error amplitude
            decay_rate_alpha = popt[1]  # How quickly error decreases
            irreducible_residual_c = popt[2]  # Plateau that cannot be eliminated
            r_fine = np.linspace(2, 30, 100)
            ax.plot(r_fine, exp_decay(r_fine, *popt), linestyle='--')
            print(f"{name} {metric}: A={amplitude_A:.2f}, alpha={decay_rate_alpha:.4f}, c={irreducible_residual_c:.3f}")
        except:
            print(f"{name} {metric}: 拟合失败")
    ax.set_xlabel("Window Radius")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/p53_mutations/three_metrics_convergence.png", dpi=150)
plt.show()
print("三度量收敛图已保存。")