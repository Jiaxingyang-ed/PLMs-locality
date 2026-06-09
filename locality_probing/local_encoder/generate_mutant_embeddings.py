#!/usr/bin/env python3

import torch, os, pickle, glob, numpy as np, pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from Bio import SeqIO
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RAW_P53_FASTA, SKEMPI_CLEAN, RESIDUE_EMB_DIR

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
DEVICE = "cpu"
MAX_LENGTH = 1024

# 1. Load precomputed WT embeddings (indexed by protein name)
print("Loading WT embeddings...)
wt_embs = {}
for f in glob.glob(os.path.join(RESIDUE_EMB_DIR, "*.pt")):
    name = os.path.basename(f).replace(".pt", "")
    wt_embs[name] = torch.load(f, map_location='cpu')

# 2. Load SKEMPI mutations
df = pd.read_csv(SKEMPI_CLEAN)  # columns: mutated_protein, wt_aa, position, mut_aa, ddG
print(f"Total mutations: {len(df)}")

# 3. Load ProtT5
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# 4. For each mutation, generate mutant sequence embedding and compute Δh
deltas = {}  # key: (protein, pos, mut) -> (Δh, local_seq_window)
for idx, row in df.iterrows():
    protein = row['mutated_protein']
    pos = int(row['position']) - 1  # 0-indexed
    wt_aa = row['wt_aa']
    mut_aa = row['mut_aa']
    
    if protein not in wt_embs: continue
    wt_emb = wt_embs[protein]   # (L, 1024)
    L = wt_emb.shape[0]
    if pos >= L: continue
    
    # get WT sequence (from the amino acid sequence used to generate the embedding)
    # We don't have raw sequence stored; we need to retrieve it. 
    # For now, we'll approximate by using a lookup from the name->sequence mapping
    # built earlier. That mapping must be saved during extraction.
    # Since we didn't save sequences, we must do it now: reload from UniProt or cache.
    # For speed, we'll assume sequences are stored in RESIDUE_EMB_DIR/sequences.pkl.
    # We'll create that file first.

print("Need sequence cache. Please run a script to save sequences first.")