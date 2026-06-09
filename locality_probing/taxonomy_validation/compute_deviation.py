#!/usr/bin/env python3
"""Compute locality deviation for all mutations and test taxonomy hypothesis."""

import torch, pickle, numpy as np, pandas as pd, os, sys
from torch import nn
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR

MODEL_PATH = "locality_probing/local_encoder/local_encoder.pt"
MAX_WINDOW_LEN = 21
DEVICE = "cpu"

# ================== 1. Model definition (same as training) ==================
class LocalSeqEncoder(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=128, output_dim=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, seq_idx, mut_idx):
        x = self.embed(seq_idx)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ================== 2. Load model ==================
model = LocalSeqEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================== 3. Load mutation data ==================
with open(os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl'), 'rb') as f:
    data = pickle.load(f)

aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

# ================== 4. Compute locality deviation for each mutation ==================
results = []
for d in data:
    local_seq = d['local_seq']
    mut_aa = d['mut_aa']
    delta_h_global = d['delta_h_global']
    # Encode sequence window
    seq_idx = [aa_to_idx.get(aa, 0) for aa in local_seq]
    if len(seq_idx) < MAX_WINDOW_LEN:
        seq_idx += [0] * (MAX_WINDOW_LEN - len(seq_idx))
    else:
        seq_idx = seq_idx[:MAX_WINDOW_LEN]
    seq_tensor = torch.tensor(seq_idx).unsqueeze(0).to(DEVICE)
    mut_tensor = torch.tensor([aa_to_idx.get(mut_aa, 0)]).to(DEVICE)
    # Predict Δh_local
    with torch.no_grad():
        delta_h_local = model(seq_tensor, mut_tensor).squeeze().cpu().numpy()
    delta_h_global = np.array(delta_h_global)
    # Compute locality deviation L = 1 - cos(Δh_local, Δh_global)
    norm_l = np.linalg.norm(delta_h_local)
    norm_g = np.linalg.norm(delta_h_global)
    if norm_l < 1e-6 or norm_g < 1e-6:
        L = 0.0
    else:
        cos = np.dot(delta_h_local, delta_h_global) / (norm_l * norm_g)
        L = 1.0 - cos
    results.append({
        'protein': d['protein'],
        'position': d['position'],
        'wt_aa': d['wt_aa'],
        'mut_aa': d['mut_aa'],
        'locality_deviation': L,
        'ddG': d.get('ddG', 0.0)
    })

df = pd.DataFrame(results)

# ================== 5. Classify mutations into groups ==================
# We use ddG as a proxy: Group A = conservative (|ddG| < 1 kcal/mol), Group B = disruptive (|ddG| >= 1)
groupA = df[df['ddG'].abs() < 1.0]['locality_deviation'].values
groupB = df[df['ddG'].abs() >= 1.0]['locality_deviation'].values

print(f"Group A (conservative, n={len(groupA)}): mean L = {np.mean(groupA):.4f}")
print(f"Group B (disruptive, n={len(groupB)}): mean L = {np.mean(groupB):.4f}")

# ================== 6. Mann-Whitney U test ==================
stat, p = mannwhitneyu(groupB, groupA, alternative='greater')
print(f"Mann-Whitney U (B > A): statistic = {stat:.3f}, p = {p:.5f}")

# ================== 7. Visualization ==================
plt.figure(figsize=(6, 5))
plt.boxplot([groupA, groupB], labels=["Conservative (A)", "Disruptive (B)"])
plt.ylabel("Locality Deviation (1 - cos)")
plt.title(f"Locality Deviation by Mutation Type (p = {p:.4f})")
plt.savefig("locality_probing/taxonomy_validation/locality_deviation_boxplot.png")
print("Boxplot saved.")