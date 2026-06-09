#!/usr/bin/env python3
"""
ESM-2 P53 Multi-Metric Analysis with NOC Correlation

This script analyzes p53 mutations using ESM-2 embeddings to extract multiple
locality metrics and correlate them with experimental conformational heterogeneity
(NOC clusters from NMR). The analysis extracts:
- Minimum L_r (optimal local recovery)
- Convergence amplitude (delta L_r)
- Overfitting at large windows
- Irreducible residual plateau (c) from partial fitting

Scientific Purpose:
To validate that the correlation between irreducible residuals and conformational
heterogeneity holds across different protein language models (ESM-2 vs ProtT5).

Dependencies:
- torch, pickle, numpy, pandas
- scipy.optimize.curve_fit (for exponential fitting)
- scipy.stats.spearmanr (for correlation analysis)
- matplotlib (for visualization)

Inputs:
- ESM-2 p53 mutation data with precomputed global embedding deltas (p53_mutation_deltas_esm2.pkl)
- Trained ESM-2 local encoder (local_encoder_esm2.pt)
- P53 NOC cluster mapping (from config.py)

Outputs:
- Multi-panel figure showing correlation between metrics and NOC clusters
- Console output of fitted parameters and correlation coefficients
- Results saved as PNG (esm2_multi_metric.png)

Analysis Process:
1. Compute L_r across different window radii for each mutation
2. Extract multiple metrics: min L_r, delta L_r, overfit, partial fit c
3. Correlate metrics with experimental NOC clusters using Spearman correlation
4. Visualize correlations in multi-panel scatter plots
"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from torch import nn
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models import LocalSeqEncoderESM
from config import (DEVICE, MAX_ANALYSIS_WINDOW, LOCAL_ENCODER_ESM2, 
                   ESM2_EMBED_DIM, ESM2_DATA_DIR, ANALYSIS_RADII, 
                   AA_TO_IDX, P53_NOC_MAP)

MAX_WINDOW_LEN = MAX_ANALYSIS_WINDOW
MODEL_PATH = LOCAL_ENCODER_ESM2
EMBED_DIM = ESM2_EMBED_DIM

model = LocalSeqEncoderESM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

aa_to_idx = AA_TO_IDX
radii = ANALYSIS_RADII

with open(os.path.join(ESM2_DATA_DIR, "p53_mutation_deltas_esm2.pkl"), 'rb') as f:
    p53_data = pickle.load(f)

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

print("=== ESM-2 p53 多指标分析 ===\n")
results = []
for sample in p53_data:
    local_seq = sample['local_seq_full']
    center = sample['center_in_local']
    dh_global = sample['delta_h_global']
    L_r_vals = []
    for r in radii:
        start = max(0, center - r)
        end = min(len(local_seq), center + r + 1)
        sub_seq = local_seq[start:end]
        seq_idx = [aa_to_idx.get(aa,0) for aa in sub_seq]
        if len(seq_idx) < MAX_WINDOW_LEN:
            seq_idx += [0]*(MAX_WINDOW_LEN-len(seq_idx))
        else:
            seq_idx = seq_idx[:MAX_WINDOW_LEN]
        seq_tensor = torch.tensor(seq_idx).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dh_local = model(seq_tensor).squeeze().cpu().numpy()
        # L_r: Residual error (normalized Euclidean distance)
        L_r = np.linalg.norm(dh_local - dh_global)/(np.linalg.norm(dh_global) + 1e-12)
        L_r_vals.append(L_r)

    # Metric 1: Minimum L_r (optimal local recovery)
    min_L_r = min(L_r_vals)
    best_r = radii[L_r_vals.index(min_L_r)]

    # Metric 2: Convergence amplitude (r=2 vs optimal)
    delta_L_r = L_r_vals[0] - min_L_r

    # Metric 3: Overfitting at large windows (r=30 vs optimal)
    overfit = L_r_vals[-1] - min_L_r

    # Metric 4: Partial fit (only first 5 points: r=2,5,10,15,20)
    try:
        popt, _ = curve_fit(exp_decay, radii[:5], L_r_vals[:5], p0=[3, 0.1, 1], maxfev=10000)
        irreducible_residual_c = popt[2]  # Plateau from partial fit
        decay_rate_alpha = popt[1]
    except:
        irreducible_residual_c = np.nan
        decay_rate_alpha = np.nan

    results.append({
        'mutation': sample['name'],
        'min_L_r': min_L_r,
        'best_r': best_r,
        'delta_L_r': delta_L_r,
        'overfit': overfit,
        'c_partial': irreducible_residual_c,
        'alpha_partial': decay_rate_alpha,
        'L_r_vals': L_r_vals,
        'NOC': None
    })

for r in results:
    r['NOC'] = P53_NOC_MAP.get(r['mutation'], np.nan)

# 打印所有指标
for r in results:
    print(f"{r['mutation']}: L_r_values={[f'{x:.3f}' for x in r['L_r_vals']]}")
    print(f"  min_L_r={r['min_L_r']:.3f} (r={r['best_r']}), delta={r['delta_L_r']:.3f}, overfit={r['overfit']:.3f}")
    if not np.isnan(r['c_partial']):
        print(f"  c_partial={r['c_partial']:.3f}, alpha_partial={r.get('alpha_partial', np.nan):.4f}")
    print()

# 计算各指标与NOC的相关性
df = pd.DataFrame(results)
print("=== 各指标与NOC的Spearman相关性 ===")
for col in ['min_L_r', 'delta_L_r', 'overfit', 'c_partial']:
    valid = df.dropna(subset=[col, 'NOC'])
    if len(valid) >= 3:
        rho, p = spearmanr(valid[col], valid['NOC'])
        print(f"  {col}: ρ = {rho:.3f}, p = {p:.4f}")

# 多面板图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics = ['min_L_r', 'delta_L_r', 'overfit', 'c_partial']
titles = ['Min L_r (optimal recovery)', 'Delta L_r (convergence)', 'Overfit (r30 - r_min)', 'c (partial fit, r≤20)']
for ax, metric, title in zip(axes.flat, metrics, titles):
    valid = df.dropna(subset=[metric, 'NOC'])
    ax.scatter(valid[metric], valid['NOC'])
    for _, row in valid.iterrows():
        ax.annotate(row['mutation'], (row[metric], row['NOC']))
    if len(valid) >= 3:
        rho, _ = spearmanr(valid[metric], valid['NOC'])
        ax.set_title(f'{title}\nρ = {rho:.3f}')
    else:
        ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel('NOC clusters')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/esm2/esm2_multi_metric.png', dpi=150)
plt.show()
print("\n图表已保存。")