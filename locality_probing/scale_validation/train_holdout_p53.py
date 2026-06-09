#!/usr/bin/env python3
"""Family Holdout：留出p53家族，重新训练局部编码器，验证c与NOC的相关性"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR

# ================== 配置 ==================
DEVICE = "cpu"
MAX_WINDOW_LEN = 61
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
LAMBDA_COS = 0.5
MODEL_SAVE_PATH = "locality_probing/local_encoder/local_encoder_holdout_p53.pt"
DATA_PATH = os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl')
P53_DATA_PATH = "processed_data/p53_mutations/p53_mutation_deltas.pkl"

# ================== 模型定义 ==================
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

# ================== 数据加载 ==================
with open(DATA_PATH, 'rb') as f:
    all_data = pickle.load(f)

# 去重
seen = {}
unique_data = []
for d in all_data:
    key = (d['protein'], d['position'], d['mut_aa'])
    if key not in seen:
        seen[key] = d
        unique_data.append(d)

# 识别p53蛋白
p53_proteins = [d['protein'] for d in unique_data if 'p53' in d['protein'].lower() or 'TP53' in d['protein']]
print(f"找到p53相关蛋白: {set(p53_proteins)}")
print(f"p53突变数: {len(p53_proteins)}")

# 排除p53
train_data = [d for d in unique_data if d['protein'] not in set(p53_proteins)]
print(f"训练集突变数: {len(train_data)}")

# 按蛋白划分训练/验证（用于早停）
proteins = list(set(d['protein'] for d in train_data))
train_prots, val_prots = train_test_split(proteins, test_size=0.2, random_state=42)
train_ds = [d for d in train_data if d['protein'] in train_prots]
val_ds = [d for d in train_data if d['protein'] in val_prots]

# ================== Dataset ==================
class DeltaDataset(Dataset):
    def __init__(self, data, max_len=MAX_WINDOW_LEN):
        self.samples = []
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        for d in data:
            seq = d['local_seq']
            idx = [aa_to_idx.get(aa, 0) for aa in seq]
            if len(idx) < max_len:
                idx += [0] * (max_len - len(idx))
            else:
                idx = idx[:max_len]
            self.samples.append((torch.tensor(idx, dtype=torch.long),
                                 torch.tensor(d['delta_h_global'], dtype=torch.float)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

train_loader = DataLoader(DeltaDataset(train_ds), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DeltaDataset(val_ds), batch_size=BATCH_SIZE)

# ================== 训练 ==================
model = LocalSeqEncoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val_loss = float('inf')
patience = 20
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for seq, target in train_loader:
        seq, target = seq.to(DEVICE), target.to(DEVICE)
        pred = model(seq)
        mse = nn.functional.mse_loss(pred, target)
        cos_sim = nn.functional.cosine_similarity(pred, target, dim=1).mean()
        loss = mse + LAMBDA_COS * (1 - cos_sim)
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
            cos_sim = nn.functional.cosine_similarity(pred, target, dim=1).mean()
            val_loss += (mse + LAMBDA_COS * (1 - cos_sim)).item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"早停于 epoch {epoch+1}")
            break
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

print("训练完成，模型已保存。")

# ================== 评估p53突变 ==================
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()
aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

with open(P53_DATA_PATH, 'rb') as f:
    p53_data = pickle.load(f)

radii = [2, 5, 10, 15, 20, 30]
def exp_decay(r, A, alpha, c):
    return A * np.exp(-alpha * r) + c

results = []
for sample in p53_data:
    local_seq = sample["local_seq_full"]
    center = sample["center_in_local"]
    dh_global = sample["delta_h_global"]
    L_r_vals = []
    for r in radii:
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
            dh_local = model(seq_tensor).squeeze().cpu().numpy()
        norm_l = np.linalg.norm(dh_local)
        norm_g = np.linalg.norm(dh_global)
        L_r = np.linalg.norm(dh_local - dh_global) / (norm_g + 1e-12)
        L_r_vals.append(L_r)
    try:
        popt, _ = curve_fit(exp_decay, radii, L_r_vals, p0=[10, 0.1, 3], maxfev=10000)
        c = popt[2]
    except:
        c = np.nan
    results.append({'mutation': sample['name'], 'c_Lr': c, 'NOC': None})

# 填入已知NOC值（来自文献）
noc_map = {'R175H': 42, 'G245S': 32, 'R249S': 30, 'R282W': 36, 'Y220C': 32}
for r in results:
    r['NOC'] = noc_map.get(r['mutation'], np.nan)

df = pd.DataFrame(results).dropna()
rho, p = spearmanr(df['c_Lr'], df['NOC'])
print(f"留出p53后，c_Lr与NOC的Spearman ρ = {rho:.3f}, p = {p:.4f}")
print(df)