#!/usr/bin/env python3
"""使用余弦校准后的局部编码器重新评估p53五突变"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR

# ================== 配置 ==================
DEVICE = "cpu"
MAX_WINDOW_LEN = 21
MODEL_PATH = "locality_probing/local_encoder/local_encoder_cosine.pt"
DATA_PATH = "processed_data/p53_mutations/p53_mutation_deltas.pkl"

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

model = LocalSeqEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

def compute_L_measures(delta_h_local, delta_h_global):
    delta_h_local = np.asarray(delta_h_local)
    delta_h_global = np.asarray(delta_h_global)
    norm_l = np.linalg.norm(delta_h_local)
    norm_g = np.linalg.norm(delta_h_global)
    if norm_g < 1e-12:
        return 0.0, 0.0, 0.0
    if norm_l < 1e-12:
        L_theta = 1.0
    else:
        cos = np.dot(delta_h_local, delta_h_global) / (norm_l * norm_g)
        cos = np.clip(cos, -1.0, 1.0)
        L_theta = 1.0 - cos
    eps = 1e-8
    L_m = np.abs(np.log((norm_l + eps) / (norm_g + eps)))
    L_r = np.linalg.norm(delta_h_local - delta_h_global) / (norm_g + eps)
    return L_theta, L_m, L_r

# 加载p53突变数据
with open(DATA_PATH, 'rb') as f:
    p53_data = pickle.load(f)

radii = [5, 10, 20]
results = []

for sample in p53_data:
    local_seq = sample["local_seq_full"]
    center = sample["center_in_local"]
    delta_h_global = sample["delta_h_global"]

    for r in radii:
        # 裁剪窗口
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
            delta_h_local = model(seq_tensor, None).squeeze().cpu().numpy()
        L_theta, L_m, L_r = compute_L_measures(delta_h_local, delta_h_global)
        results.append({
            "mutation": sample["name"],
            "category": sample["category"],
            "radius": r,
            "L_theta": L_theta,
            "L_m": L_m,
            "L_r": L_r
        })
        print(f"{sample['name']} r={r}: L_θ={L_theta:.4f}, L_m={L_m:.4f}, L_r={L_r:.4f}")

df = pd.DataFrame(results)
df.to_csv("processed_data/p53_mutations/p53_locality_measures_cosine.csv", index=False)
print("\n结果已保存。")