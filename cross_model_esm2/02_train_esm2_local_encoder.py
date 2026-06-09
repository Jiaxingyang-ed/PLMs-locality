#!/usr/bin/env python3
"""
Train ESM-2 Local Sequence Encoder for Cross-Model Validation

This script trains a local sequence encoder using ESM-2 embeddings (640-dimensional)
to enable cross-model validation of locality findings. The training process uses
a combined loss function (MSE + cosine similarity) to ensure both magnitude and
directional accuracy.

Scientific Purpose:
To validate that locality convergence dynamics are consistent across different
protein language models (ProtT5 vs ESM-2), strengthening the generality of findings.

Dependencies:
- torch, pickle, numpy
- torch.utils.data.DataLoader
- sklearn.model_selection.train_test_split
- tqdm (for progress bars)

Inputs:
- ESM-2 mutation data with precomputed global embedding deltas (mutation_deltas_esm2.pkl)
- Configuration parameters from config.py

Outputs:
- Trained ESM-2 model weights (local_encoder_esm2.pt)
- Training/validation loss curves (console output)
- Best model checkpoint based on validation loss

Training Process:
1. Load ESM-2 mutation data with local sequences and global embedding deltas
2. Split proteins into train/validation sets (by protein, not by mutation)
3. Train transformer encoder with combined MSE + cosine loss
4. Save best model based on validation loss with early stopping
"""

import torch, pickle, numpy as np, os, sys
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models import LocalSeqEncoderESM, DeltaDataset
from config import (ESM2_EMBED_DIM, ESM2_WINDOW_LEN, ESM2_BATCH_SIZE, 
                   ESM2_EPOCHS, ESM2_LEARNING_RATE, ESM2_LAMBDA_COS, 
                   DEVICE, ESM2_DATA_DIR, LOCAL_ENCODER_ESM2, 
                   VAL_TEST_SPLIT, RANDOM_STATE)

EMBED_DIM = ESM2_EMBED_DIM
MAX_WINDOW_LEN = ESM2_WINDOW_LEN
BATCH_SIZE = ESM2_BATCH_SIZE
EPOCHS = ESM2_EPOCHS
LR = ESM2_LEARNING_RATE
LAMBDA_COS = ESM2_LAMBDA_COS

with open(os.path.join(ESM2_DATA_DIR, "mutation_deltas_esm2.pkl"), 'rb') as f:
    all_data = pickle.load(f)

proteins = list(set(d['protein'] for d in all_data))
train_prots, val_prots = train_test_split(proteins, test_size=VAL_TEST_SPLIT, random_state=RANDOM_STATE)
train_data = [d for d in all_data if d['protein'] in train_prots]
val_data   = [d for d in all_data if d['protein'] in val_prots]

train_loader = DataLoader(DeltaDataset(train_data, max_len=MAX_WINDOW_LEN, include_mut_idx=False), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(DeltaDataset(val_data, max_len=MAX_WINDOW_LEN, include_mut_idx=False), batch_size=BATCH_SIZE)

model = LocalSeqEncoderESM().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val = float('inf')
patience = 20
no_imp = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for seq, target in train_loader:
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        pred = model(seq)
        # Combined loss: MSE (magnitude) + cosine similarity (direction)
        mse = nn.functional.mse_loss(pred, target)
        cos = nn.functional.cosine_similarity(pred, target, dim=1).mean()
        loss = mse + LAMBDA_COS*(1-cos)  # L = MSE + λ * (1 - cos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for seq, target in val_loader:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            pred = model(seq)
            mse = nn.functional.mse_loss(pred, target)
            cos = nn.functional.cosine_similarity(pred, target, dim=1).mean()
            val_loss += (mse + LAMBDA_COS*(1-cos)).item()
    val_loss /= len(val_loader)

    if val_loss < best_val:
        best_val = val_loss
        no_imp = 0
        torch.save(model.state_dict(), LOCAL_ENCODER_ESM2)
    else:
        no_imp += 1
        if no_imp >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    if (epoch+1)%20==0:
        print(f"Epoch {epoch+1}: train {train_loss:.3f} val {val_loss:.3f}")