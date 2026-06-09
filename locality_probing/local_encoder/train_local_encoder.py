#!/usr/bin/env python3
"""
Train a Local Sequence Encoder to Predict Global Embedding Deltas

This script trains a local sequence encoder to predict mutation-induced changes in
global residue embeddings (Δh) from local sequence windows. The model learns to
capture local context that can approximate global embedding changes.

Scientific Purpose:
To quantify how much of the global embedding change can be explained by local
sequence context, establishing a baseline for locality analysis.

Dependencies:
- torch, pickle, numpy
- torch.utils.data.DataLoader
- sklearn.model_selection.train_test_split
- tqdm (for progress bars)

Inputs:
- Mutation data with precomputed global embedding deltas (mutation_deltas.pkl)
- Configuration parameters from config.py

Outputs:
- Trained model weights (local_encoder.pt)
- Training/validation loss curves (console output)
- Best model checkpoint based on validation loss

Training Process:
1. Load mutation data with local sequences and global embedding deltas
2. Split proteins into train/validation sets (by protein, not by mutation)
3. Train transformer encoder to predict Δh from local sequence windows
4. Save best model based on validation loss
"""

import torch, os, pickle, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models import LocalSeqEncoder, DeltaDataset
from config import (RESIDUE_EMB_DIR, PROTT5_BATCH_SIZE, PROTT5_EPOCHS, 
                   PROTT5_LEARNING_RATE, PROTT5_WINDOW_LEN, DEVICE,
                   LOCAL_ENCODER_PROTT5, TRAIN_TEST_SPLIT, RANDOM_STATE)

# ================== Config ==================
BATCH_SIZE = PROTT5_BATCH_SIZE
EPOCHS = PROTT5_EPOCHS
LEARNING_RATE = PROTT5_LEARNING_RATE
MODEL_SAVE_PATH = LOCAL_ENCODER_PROTT5
MAX_WINDOW_LEN = PROTT5_WINDOW_LEN

# ================== 3. Training ==================
def train():
    with open(os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl'), 'rb') as f:
        data = pickle.load(f)

    proteins = list(set(d['protein'] for d in data))
    train_prots, val_prots = train_test_split(proteins, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)
    train_data = [d for d in data if d['protein'] in train_prots]
    val_data = [d for d in data if d['protein'] in val_prots]

    train_ds = DeltaDataset(train_data)
    val_ds = DeltaDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LocalSeqEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for seq_idx, mut_idx, delta_h in train_loader:
            seq_idx, mut_idx, delta_h = seq_idx.to(DEVICE), mut_idx.to(DEVICE), delta_h.to(DEVICE)
            pred = model(seq_idx, mut_idx)
            loss = loss_fn(pred, delta_h)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq_idx, mut_idx, delta_h in val_loader:
                seq_idx, mut_idx, delta_h = seq_idx.to(DEVICE), mut_idx.to(DEVICE), delta_h.to(DEVICE)
                pred = model(seq_idx, mut_idx)
                val_loss += loss_fn(pred, delta_h).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: train loss {train_loss:.6f}, val loss {val_loss:.6f}")

    print(f"Best val loss: {best_val_loss:.6f}, model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()