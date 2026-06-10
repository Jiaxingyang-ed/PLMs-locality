#!/usr/bin/env python3
"""
扩展规模验证：从SKEMPI数据集中筛选100个突变，计算三度量L(r)收敛参数。
"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from torch import nn
from scipy.optimize import curve_fit
from tqdm import tqdm

# 确保项目根目录在sys.path中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RESIDUE_EMB_DIR, LOCAL_ENCODER_COSINE
from models import LocalSeqEncoder

DEVICE = "cpu"
MAX_WINDOW_LEN = 61
MODEL_PATH = LOCAL_ENCODER_COSINE

model = LocalSeqEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

def compute_all_L(dh_local, dh_global):
    norm_l = np.linalg.norm(dh_local)
    norm_g = np.linalg.norm(dh_global)
    eps = 1e-12
    if norm_g < eps: return 0.0, 0.0, 0.0
    if norm_l < eps: L_theta = 1.0
    else:
        cos = np.dot(dh_local, dh_global)/(norm_l*norm_g)
        L_theta = 1.0 - np.clip(cos, -1, 1)
    L_m = np.abs(np.log((norm_l+eps)/(norm_g+eps)))
    L_r = np.linalg.norm(dh_local - dh_global)/(norm_g+eps)
    return L_theta, L_m, L_r

# ================== 加载并筛选突变 ==================
with open(os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl'), 'rb') as f:
    all_data = pickle.load(f)

# 去重
seen = {}
for d in all_data:
    key = (d['protein'], d['position'], d['mut_aa'])
    if key not in seen:
        seen[key] = d
unique_data = list(seen.values())
print(f"去重后突变总数: {len(unique_data)}")

# 筛选：优先覆盖多个蛋白家族，每个蛋白至少保留3个突变
protein_counts = {}
for d in unique_data:
    protein_counts[d['protein']] = protein_counts.get(d['protein'], 0) + 1

selected = []
proteins_with_enough = [p for p, c in protein_counts.items() if c >= 3]
print(f"有≥3个突变的蛋白数: {len(proteins_with_enough)}")

# 从每个蛋白中随机选最多5个突变（避免某个蛋白主导）
np.random.seed(42)
for prot in proteins_with_enough:
    prot_muts = [d for d in unique_data if d['protein'] == prot]
    n_select = min(5, len(prot_muts))
    selected.extend(np.random.choice(prot_muts, n_select, replace=False))

if len(selected) > 100:
    selected = np.random.choice(selected, 100, replace=False).tolist()
elif len(selected) < 100:
    # 如果不够100，从剩余突变中随机补充
    remaining = [d for d in unique_data if d not in selected]
    needed = 100 - len(selected)
    selected.extend(np.random.choice(remaining, needed, replace=False).tolist())

print(f"最终选取突变数: {len(selected)}")
print(f"覆盖蛋白数: {len(set(d['protein'] for d in selected))}")

# ================== 计算L(r)并拟合 ==================
radii = [2, 5, 10, 15, 20, 30]

def exp_decay(r, A, alpha, c):
    return A * np.exp(-alpha * r) + c

results = []
for sample in tqdm(selected):
    local_seq = sample['local_seq']
    center = len(local_seq) // 2  # 突变残基在窗口中心
    dh_global = sample['delta_h_global']
    L_theta_vals, L_m_vals, L_r_vals = [], [], []
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
        L_theta_vals.append(L_theta)
        L_m_vals.append(L_m)
        L_r_vals.append(L_r)

    # 拟合指数衰减（主要对L_r，因为它是最综合的度量）
    try:
        popt_r, _ = curve_fit(exp_decay, radii, L_r_vals, p0=[10, 0.1, 3], maxfev=10000)
        A_r, alpha_r, c_r = popt_r
    except:
        A_r, alpha_r, c_r = np.nan, np.nan, np.nan

    try:
        popt_m, _ = curve_fit(exp_decay, radii, L_m_vals, p0=[2, 0.07, 1], maxfev=10000)
        A_m, alpha_m, c_m = popt_m
    except:
        A_m, alpha_m, c_m = np.nan, np.nan, np.nan

    results.append({
        'protein': sample['protein'],
        'wt_aa': sample['wt_aa'],
        'position': sample['position'],
        'mut_aa': sample['mut_aa'],
        'ddG': sample.get('ddG', np.nan),
        'c_Lr': c_r, 'alpha_Lr': alpha_r,
        'c_Lm': c_m, 'alpha_Lm': alpha_m,
        'L_theta_r2': L_theta_vals[0],  # r=2时的L_θ作为基线
        'L_theta_r30': L_theta_vals[-1],  # r=30时的L_θ
    })

df = pd.DataFrame(results)
os.makedirs("processed_data/scale_validation", exist_ok=True)
df.to_csv("processed_data/scale_validation/100_mutations_convergence_params.csv", index=False)
print(f"保存完成。收敛参数c_Lr的范围: {df['c_Lr'].min():.2f} - {df['c_Lr'].max():.2f}")
print(f"跨蛋白的c_Lr标准差: {df['c_Lr'].std():.3f}")
print(f"c_Lr与ddG的Spearman相关系数: {df['c_Lr'].corr(df['ddG'], method='spearman'):.3f}")