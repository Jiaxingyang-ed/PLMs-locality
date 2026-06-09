#!/usr/bin/env python3
"""为 SKEMPI 402 个突变和 5 个 p53 突变生成 ESM-2 (150M) 的 Δh 和局部序列"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
from Bio import SeqIO

# ---------- 确保能找到 config ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "scripts"))
from config import RESIDUE_EMB_DIR, RAW_P53_FASTA

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
DEVICE = "cpu"
MAX_LEN = 1024

print("加载 ESM-2 150M...")
tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def embed_sequence(seq):
    seq = seq[:MAX_LEN]
    inputs = tokenizer(seq, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.last_hidden_state[0, 1:-1, :].cpu()   # (L, 640)
    return hidden

# ---------- 1. SKEMPI 402 突变 ----------
with open(os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl'), 'rb') as f:
    skempi_data = pickle.load(f)

with open(os.path.join(RESIDUE_EMB_DIR, 'sequences.pkl'), 'rb') as f:
    sequences = pickle.load(f)

new_data = []
for d in tqdm(skempi_data, desc="Processing SKEMPI mutations"):
    prot = d['protein']
    if prot not in sequences:
        continue
    seq = sequences[prot]
    pos = d['position']
    wt = d['wt_aa']
    mut = d['mut_aa']
    if pos >= len(seq) or seq[pos] != wt:
        continue
    mut_seq = seq[:pos] + mut + seq[pos+1:]
    wt_emb = embed_sequence(seq)
    mut_emb = embed_sequence(mut_seq)
    if pos >= wt_emb.shape[0] or pos >= mut_emb.shape[0]:
        continue
    delta_h = (mut_emb[pos] - wt_emb[pos]).numpy()
    radius = 10
    start = max(0, pos - radius)
    end = min(len(seq), pos + radius + 1)
    local_seq = seq[start:end]
    new_data.append({
        'protein': prot, 'position': pos, 'wt_aa': wt, 'mut_aa': mut,
        'local_seq': local_seq, 'delta_h_global': delta_h,
        'ddG': d.get('ddG', 0.0)
    })

os.makedirs("processed_data/esm2", exist_ok=True)
with open("processed_data/esm2/mutation_deltas_esm2.pkl", 'wb') as f:
    pickle.dump(new_data, f)
print(f"Saved {len(new_data)} SKEMPI mutations for ESM-2.")

# ---------- 2. p53 五个突变 ----------
with open(RAW_P53_FASTA) as f:
    p53_seq = str(next(SeqIO.parse(f, "fasta")).seq)

p53_mutations = [
    ("R175H", 174, "R", "H"),
    ("G245S", 244, "G", "S"),
    ("R249S", 248, "R", "S"),
    ("R282W", 281, "R", "W"),
    ("Y220C", 219, "Y", "C"),
]

wt_emb = embed_sequence(p53_seq)
p53_data = []
for name, pos_0idx, wt, mut in p53_mutations:
    mut_seq = p53_seq[:pos_0idx] + mut + p53_seq[pos_0idx+1:]
    mut_emb = embed_sequence(mut_seq)
    delta_h = (mut_emb[pos_0idx] - wt_emb[pos_0idx]).numpy()
    radius = 20
    start = max(0, pos_0idx - radius)
    end = min(len(p53_seq), pos_0idx + radius + 1)
    local_seq_full = p53_seq[start:end]
    center_in_local = pos_0idx - start
    p53_data.append({
        'name': name, 'category': 'global' if name in ('R175H','R282W','Y220C') else 'local',
        'position': pos_0idx, 'wt_aa': wt, 'mut_aa': mut,
        'local_seq_full': local_seq_full, 'center_in_local': center_in_local,
        'delta_h_global': delta_h
    })

with open("processed_data/esm2/p53_mutation_deltas_esm2.pkl", 'wb') as f:
    pickle.dump(p53_data, f)
print("Saved p53 mutations for ESM-2.")