#!/usr/bin/env python3
"""Lightweight ESM-2 contact change calculation (memory-safe)."""

import torch, pickle, numpy as np, os, sys, pandas as pd
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEVICE = "cpu"
MAX_LEN = 200  # 截断长序列，大幅减少内存

# ---------- 加载模型 ----------
print("加载 ESM-2 8M...")
tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME, output_attentions=True).to(DEVICE)
model.eval()

# ---------- 加载序列 ----------
with open(os.path.join(RESIDUE_EMB_DIR, 'sequences.pkl'), 'rb') as f:
    sequences = pickle.load(f)

# ---------- 金标准突变 ----------
mutations = [
    ('Bovine alpha-chymotrypsin', 9, 'I', 'D', 'disruptive'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'E', 'disruptive'),
    ('Subtilisin BPN', 41, 'I', 'P', 'disruptive'),
    ('Subtilisin Carlsberg', 41, 'I', 'P', 'disruptive'),
    ('Human leukocyte elastase', 30, 'I', 'R', 'disruptive'),
    ('Subtilisin BPN', 41, 'I', 'G', 'disruptive'),
    ('HyHEL-10', 34, 'L', 'A', 'disruptive'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'P', 'disruptive'),
    ('Human leukocyte elastase', 30, 'I', 'M', 'disruptive'),
    ('Bovine trypsin', 5, 'I', 'A', 'disruptive'),
    ('Subtilisin BPN', 41, 'I', 'M', 'conservative'),
    ('Factor VIIa', 56, 'T', 'A', 'conservative'),
    ('Chemotaxis protein CheY', 88, 'A', 'V', 'conservative'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'L', 'conservative'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'G', 'conservative'),
    ('Streptomyces griseus proteinase B', 3, 'I', 'A', 'conservative'),
    ('Subtilisin BPN', 41, 'I', 'Y', 'conservative'),
    ('Chemotaxis protein CheY', 48, 'A', 'C', 'conservative'),
]

def predict_contact_change(seq_wt, seq_mut, pos):
    """Calculate local contact change around the mutation site."""
    # 截断到 MAX_LEN
    seq_wt = seq_wt[:MAX_LEN]
    seq_mut = seq_mut[:MAX_LEN]
    # Tokenize
    inputs_wt = tokenizer(seq_wt, return_tensors="pt").to(DEVICE)
    inputs_mut = tokenizer(seq_mut, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_wt = model(**inputs_wt, output_attentions=True)
        out_mut = model(**inputs_mut, output_attentions=True)
    # 取最后一层注意力 (1, heads, L, L) → 平均头 → (L, L)
    attn_wt = out_wt.attentions[-1].mean(dim=1).squeeze(0).cpu().numpy()
    attn_mut = out_mut.attentions[-1].mean(dim=1).squeeze(0).cpu().numpy()
    # 计算突变位点所在行（接触）的变化
    pos_token = pos + 1  # token position (加 <cls>)
    if pos_token >= attn_wt.shape[0] or pos_token >= attn_mut.shape[0]:
        return 0.0
    diff = np.mean((attn_wt[pos_token] - attn_mut[pos_token])**2)
    return float(diff)

results = []
for prot, pos_1idx, wt, mt, orig_group in tqdm(mutations):
    if prot not in sequences:
        continue
    seq = sequences[prot]
    pos = pos_1idx - 1
    if pos < 0 or pos >= len(seq) or seq[pos] != wt:
        continue
    mut_seq = seq[:pos] + mt + seq[pos+1:]
    change = predict_contact_change(seq, mut_seq, pos)
    results.append({
        'protein': prot,
        'mutation': f"{wt}{pos_1idx}{mt}",
        'contact_change': change,
        'original_group': orig_group
    })
    print(f"  {wt}{pos_1idx}{mt}: local contact change = {change:.6f}")

df = pd.DataFrame(results)
os.makedirs('locality_probing/structure_analysis', exist_ok=True)
df.to_csv('locality_probing/structure_analysis/contact_change_scores.csv', index=False)
print(f"保存 {len(results)} 条结果")