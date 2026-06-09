#!/usr/bin/env python3
"""ESM-2 p53分析 - 提取L_r最小值 + 分区域拟合"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from torch import nn
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

DEVICE = "cpu"
MAX_WINDOW_LEN = 61
MODEL_PATH = "locality_probing/local_encoder/local_encoder_esm2.pt"
EMBED_DIM = 640

class LocalSeqEncoderESM(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=128, output_dim=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                                   dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, output_dim)
    def forward(self, seq_idx, mut_idx=None):
        x = self.embed(seq_idx)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

model = LocalSeqEncoderESM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

aa_to_idx = {aa:i for i,aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
radii = [2,5,10,15,20,30]

with open("processed_data/esm2/p53_mutation_deltas_esm2.pkl", 'rb') as f:
    p53_data = pickle.load(f)

def exp_decay(r, A, alpha, c):
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
        L_r = np.linalg.norm(dh_local - dh_global)/(np.linalg.norm(dh_global) + 1e-12)
        L_r_vals.append(L_r)

    # 指标1: L_r最小值（最优局部恢复）
    min_L_r = min(L_r_vals)
    best_r = radii[L_r_vals.index(min_L_r)]

    # 指标2: 收敛幅度 (r=2 vs 最优)
    delta_L_r = L_r_vals[0] - min_L_r

    # 指标3: 大窗口过拟合程度 (r=30 vs 最优)
    overfit = L_r_vals[-1] - min_L_r

    # 指标4: 仅拟合前五个点（r=2,5,10,15,20）
    try:
        popt, _ = curve_fit(exp_decay, radii[:5], L_r_vals[:5], p0=[3, 0.1, 1], maxfev=10000)
        c_partial = popt[2]
        alpha_partial = popt[1]
    except:
        c_partial = np.nan
        alpha_partial = np.nan

    results.append({
        'mutation': sample['name'],
        'min_L_r': min_L_r,
        'best_r': best_r,
        'delta_L_r': delta_L_r,
        'overfit': overfit,
        'c_partial': c_partial,
        'alpha_partial': alpha_partial,
        'L_r_vals': L_r_vals,
        'NOC': None
    })

noc_map = {'R175H':42, 'G245S':32, 'R249S':30, 'R282W':36, 'Y220C':32}
for r in results:
    r['NOC'] = noc_map.get(r['mutation'], np.nan)

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
plt.savefig('processed_data/esm2/esm2_multi_metric.png', dpi=150)
plt.show()
print("\n图表已保存。")
