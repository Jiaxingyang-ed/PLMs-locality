#!/usr/bin/env python3
"""Multi-window scale analysis using existing LocalSeqEncoder (2-layer)."""

import torch, pickle, numpy as np, os, sys
from torch import nn
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR

# ================== 旧版模型 (2层) ==================
MAX_WINDOW_LEN = 21  # 对应 r=10
DEVICE = "cpu"

class LocalSeqEncoder(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=128, output_dim=1024):
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

# ================== 加载模型 ==================
MODEL_PATH = "locality_probing/local_encoder/local_encoder.pt"
model = LocalSeqEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================== 加载并去重数据 ==================
with open(os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl'), 'rb') as f:
    data = pickle.load(f)

seen = {}
unique_data = []
for d in data:
    key = (d['protein'], d['position'], d['mut_aa'])
    if key not in seen:
        seen[key] = d
        unique_data.append(d)
print(f"唯一突变数: {len(unique_data)}")

aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

def compute_L_for_window(sample, radius):
    """计算指定窗口半径下的局部性偏差 L (填充到 MAX_WINDOW_LEN)"""
    seq = sample['local_seq']
    center = len(seq) // 2
    start = max(0, center - radius)
    end = min(len(seq), center + radius + 1)
    sub_seq = seq[start:end]

    seq_idx = [aa_to_idx.get(aa, 0) for aa in sub_seq]
    # 填充到固定长度21
    if len(seq_idx) < MAX_WINDOW_LEN:
        seq_idx += [0] * (MAX_WINDOW_LEN - len(seq_idx))
    else:
        seq_idx = seq_idx[:MAX_WINDOW_LEN]

    seq_tensor = torch.tensor(seq_idx).unsqueeze(0).to(DEVICE)
    delta_h_global = sample['delta_h_global']
    with torch.no_grad():
        delta_h_local = model(seq_tensor, None).squeeze().cpu().numpy()

    norm_l = np.linalg.norm(delta_h_local)
    norm_g = np.linalg.norm(delta_h_global)
    if norm_l < 1e-6 or norm_g < 1e-6:
        return 0.0
    cos = np.dot(delta_h_local, delta_h_global) / (norm_l * norm_g)
    return 1.0 - cos

# ================== 金标准突变列表 ==================
conform_muts = [
    ('Bovine alpha-chymotrypsin', 9, 'I', 'D'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'E'),
    ('Subtilisin BPN', 41, 'I', 'P'),
    ('Subtilisin Carlsberg', 41, 'I', 'P'),
    ('Human leukocyte elastase', 30, 'I', 'R'),
    ('Subtilisin BPN', 41, 'I', 'G'),
    ('HyHEL-10', 34, 'L', 'A'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'P'),
    ('Human leukocyte elastase', 30, 'I', 'M'),
    ('Bovine trypsin', 5, 'I', 'A'),
]
conserv_muts = [
    ('Subtilisin BPN', 41, 'I', 'M'),
    ('Factor VIIa', 56, 'T', 'A'),
    ('Chemotaxis protein CheY', 88, 'A', 'V'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'L'),
    ('HyHEL-10', 34, 'H', 'A'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'G'),
    ('Streptomyces griseus proteinase B', 3, 'I', 'A'),
    ('Factor VIIa', 16, 'T', 'A'),
    ('Subtilisin BPN', 41, 'I', 'Y'),
    ('Chemotaxis protein CheY', 48, 'A', 'C'),
]

# ================== 多窗口计算 ==================
windows = [2, 5, 10]   # 受限于旧模型窗口长度21 (r=10)
results = {r: {'conf': [], 'cons': []} for r in windows}
used_conf, used_cons = set(), set()

for r in windows:
    conf_L, cons_L = [], []
    used_conf.clear()
    used_cons.clear()
    for d in unique_data:
        prot, pos, wt, mut = d['protein'], d['position'], d['wt_aa'], d['mut_aa']
        for c in conform_muts:
            key = (c[0], c[1]-1, c[3])
            if key in used_conf: continue
            if c[0] in prot and pos == c[1]-1 and wt == c[2] and mut == c[3]:
                used_conf.add(key)
                conf_L.append(compute_L_for_window(d, r))
                break
        for c in conserv_muts:
            key = (c[0], c[1]-1, c[3])
            if key in used_cons: continue
            if c[0] in prot and pos == c[1]-1 and wt == c[2] and mut == c[3]:
                used_cons.add(key)
                cons_L.append(compute_L_for_window(d, r))
                break
    results[r]['conf'] = conf_L
    results[r]['cons'] = cons_L
    print(f"Window r={r}: Disruptive mean L={np.mean(conf_L):.3f}, Conservative mean L={np.mean(cons_L):.3f}")
    if len(conf_L)>=3 and len(cons_L)>=3:
        _, p = mannwhitneyu(conf_L, cons_L, alternative='greater')
        print(f"  Mann-Whitney p = {p:.4f}")

# ================== 可视化 ==================
plt.figure(figsize=(8,5))
for group, label, marker in [('conf', 'Disruptive', 'o'), ('cons', 'Conservative', 's')]:
    means = [np.mean(results[r][group]) for r in windows]
    plt.plot(windows, means, marker=marker, label=label)
plt.xlabel('Window Radius (residues)')
plt.ylabel('Mean Locality Deviation L')
plt.title('Scale Dependence of Locality Deviation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('locality_probing/taxonomy_validation/window_scale_analysis.png')
print("图表已保存。")