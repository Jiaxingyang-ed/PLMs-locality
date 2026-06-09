#!/usr/bin/env python3
"""Generate Δh for ALL possible SKEMPI mutations using cached sequences."""

import torch, os, pickle, numpy as np, pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import SKEMPI_CLEAN, RESIDUE_EMB_DIR

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
DEVICE = "cpu"
MAX_LENGTH = 1024

# 1. Load sequence cache
seq_path = os.path.join(RESIDUE_EMB_DIR, 'sequences.pkl')
with open(seq_path, 'rb') as f:
    sequences = pickle.load(f)
print(f"Loaded {len(sequences)} sequences")

# 2. Load SKEMPI mutations
df = pd.read_csv(SKEMPI_CLEAN)

# 3. Load ProtT5
print("Loading ProtT5...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def embed_sequence(seq):
    seq = seq[:MAX_LENGTH]
    seq_spaced = " ".join(list(seq))
    inputs = tokenizer(seq_spaced, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 1:-1, :].cpu()

# 4. Process mutations
data = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    protein = row['mutated_protein']
    if protein not in sequences:
        continue
    seq = sequences[protein]
    pos = int(row['position']) - 1  # 0-indexed
    wt_aa = row['wt_aa']
    mut_aa = row['mut_aa']
    if pos < 0 or pos >= len(seq):
        continue
    # Optionally verify wt_aa, but skip if mismatch (can happen due to numbering)
    if seq[pos] != wt_aa:
        # Try to find the correct position by scanning nearby
        found = False
        for offset in range(-5, 6):
            new_pos = pos + offset
            if 0 <= new_pos < len(seq) and seq[new_pos] == wt_aa:
                pos = new_pos
                found = True
                break
        if not found:
            continue   # cannot locate mutation

    mut_seq = seq[:pos] + mut_aa + seq[pos+1:]

    # Global embeddings
    wt_emb = embed_sequence(seq)
    mut_emb = embed_sequence(mut_seq)
    if pos >= wt_emb.shape[0] or pos >= mut_emb.shape[0]:
        continue
    delta_h_global = (mut_emb[pos] - wt_emb[pos]).numpy()

    # Local sequence window
    radius = 10
    start = max(0, pos - radius)
    end = min(len(seq), pos + radius + 1)
    local_seq = seq[start:end]

    data.append({
        'protein': protein,
        'position': pos,
        'wt_aa': wt_aa,
        'mut_aa': mut_aa,
        'local_seq': local_seq,
        'delta_h_global': delta_h_global,
        'ddG': row['ddG'] if 'ddG' in row else 0.0
    })

output_path = os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
print(f"Saved {len(data)} Δh samples to {output_path}")