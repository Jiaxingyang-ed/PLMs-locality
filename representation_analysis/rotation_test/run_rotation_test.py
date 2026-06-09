#!/usr/bin/env python3
"""
Random Rotation Test: 使用真实的AAindex疏水性标度，验证ProtT5嵌入中物化信号的非平凡性。
"""

import torch, pickle, numpy as np, os, sys, glob
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR

# ================== 配置 ==================
EMB_DIR = RESIDUE_EMB_DIR
SEQ_PATH = os.path.join(RESIDUE_EMB_DIR, 'sequences.pkl')
N_MAX_RESIDUES = 5000        # 最大采样残基数
N_ROTATIONS = 30             # 旋转次数
RANDOM_SEED = 42

# AAindex 疏水性标度 (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# ================== 1. 加载序列并收集残基嵌入 ==================
print("加载序列...")
with open(SEQ_PATH, 'rb') as f:
    sequences = pickle.load(f)

print("收集残基嵌入...")
all_embeddings = []
all_labels = []

pt_files = glob.glob(os.path.join(EMB_DIR, "*.pt"))
np.random.seed(RANDOM_SEED)
np.random.shuffle(pt_files)

for pt_file in pt_files:
    if len(all_embeddings) >= N_MAX_RESIDUES:
        break
    name = os.path.basename(pt_file).replace(".pt", "")
    if name not in sequences:
        continue
    seq = sequences[name]
    emb = torch.load(pt_file, map_location='cpu')  # (L, 1024)
    L = min(len(seq), emb.shape[0])
    # 采样（避免整个蛋白的残基高度相关）
    n_sample = min(200, L)
    indices = np.random.choice(L, n_sample, replace=False)
    for i in indices:
        aa = seq[i]
        if aa in HYDROPHOBICITY:
            all_embeddings.append(emb[i].numpy())
            all_labels.append(HYDROPHOBICITY[aa])

X = np.array(all_embeddings)
y = np.array(all_labels)
print(f"采样残基数: {len(y)}")
print(f"标签范围: [{y.min():.1f}, {y.max():.1f}]")

# ================== 2. 原始空间探针 ==================
probe = Ridge(alpha=1.0)
scores_orig = cross_val_score(probe, X, y, cv=5, scoring='r2')
r2_orig = scores_orig.mean()
print(f"原始空间 R²: {r2_orig:.4f} (±{scores_orig.std():.4f})")

# ================== 3. 旋转空间探针 ==================
D = X.shape[1]
r2_rotated = []

print(f"执行 {N_ROTATIONS} 次随机旋转...")
for _ in tqdm(range(N_ROTATIONS)):
    Q = special_ortho_group.rvs(D)
    X_rot = X @ Q.T
    scores_rot = cross_val_score(probe, X_rot, y, cv=5, scoring='r2')
    r2_rotated.append(scores_rot.mean())

r2_rot_mean = np.mean(r2_rotated)
r2_rot_std = np.std(r2_rotated)
print(f"旋转后平均 R²: {r2_rot_mean:.4f} (±{r2_rot_std:.4f})")

# ================== 4. 统计检验 ==================
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(r2_rotated, scores_orig)
pct_drop = (r2_orig - r2_rot_mean) / r2_orig * 100
print(f"R² 降幅: {pct_drop:.1f}%")
print(f"t检验 p = {p_value:.6f}")

# ================== 5. 可视化 ==================
plt.figure(figsize=(8, 5))
plt.hist(r2_rotated, bins=15, alpha=0.7, label='Rotated spaces')
plt.axvline(r2_orig, color='red', linestyle='--', linewidth=2,
            label=f'Original: {r2_orig:.3f}')
plt.axvline(r2_rot_mean, color='blue', linestyle='-', linewidth=2,
            label=f'Rotated mean: {r2_rot_mean:.3f}')
plt.xlabel('R²')
plt.ylabel('Frequency')
plt.title(f'Random Rotation Test (ΔR² = {pct_drop:.1f}%)')
plt.legend()
plt.grid(True, alpha=0.3)
os.makedirs("representation_analysis/rotation_test", exist_ok=True)
plt.savefig("representation_analysis/rotation_test/rotation_test_result.png", dpi=150)
plt.show()

print(f"\n完成。R²降幅 = {pct_drop:.1f}%")