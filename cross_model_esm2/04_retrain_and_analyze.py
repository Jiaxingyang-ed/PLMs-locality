#!/usr/bin/env python3
"""彻底重新训练ESM-2局部编码器（充分训练）并自动分析p53"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================== 配置 ==================
EMBED_DIM = 640           # ESM-2 150M 隐层维度
MAX_WINDOW_LEN = 61       # 支持 r=30
BATCH_SIZE = 32
EPOCHS = 200              # 充分训练
LR = 1e-3
LAMBDA_COS = 0.5          # 与ProtT5一致
PATIENCE = 30             # 早停耐心
DEVICE = "cpu"
MODEL_SAVE_PATH = "locality_probing/local_encoder/local_encoder_esm2_v2.pt"
DATA_PATH = "processed_data/esm2/mutation_deltas_esm2.pkl"
P53_DATA_PATH = "processed_data/esm2/p53_mutation_deltas_esm2.pkl"

# ================== 模型定义 ==================
class LocalSeqEncoderESM(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=128, output_dim=EMBED_DIM):
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

# ================== 数据集 ==================
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

# ================== 训练函数 ==================
def train():
    # 加载数据
    with open(DATA_PATH, 'rb') as f:
        all_data = pickle.load(f)
    print(f"总突变数: {len(all_data)}")

    # 蛋白级划分（与ProtT5完全一致，random_state=42）
    proteins = list(set(d['protein'] for d in all_data))
    train_prots, val_prots = train_test_split(proteins, test_size=0.2, random_state=42)
    train_data = [d for d in all_data if d['protein'] in train_prots]
    val_data = [d for d in all_data if d['protein'] in val_prots]
    print(f"训练蛋白: {len(train_prots)}, 验证蛋白: {len(val_prots)}")
    print(f"训练样本: {len(train_data)}, 验证样本: {len(val_data)}")

    train_loader = DataLoader(DeltaDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DeltaDataset(val_data), batch_size=BATCH_SIZE)

    # 模型
    model = LocalSeqEncoderESM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    patience_counter = 0

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for seq, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            pred = model(seq)
            mse = nn.functional.mse_loss(pred, target)
            cos_sim = nn.functional.cosine_similarity(pred, target, dim=1).mean()
            loss = mse + LAMBDA_COS * (1.0 - cos_sim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(DEVICE), target.to(DEVICE)
                pred = model(seq)
                mse = nn.functional.mse_loss(pred, target)
                cos_sim = nn.functional.cosine_similarity(pred, target, dim=1).mean()
                val_loss += (mse + LAMBDA_COS * (1.0 - cos_sim)).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 模型已保存 (best val = {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停于 epoch {epoch+1}")
                break

    print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")

    # 返回模型
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    return model

# ================== 分析p53 ==================
def analyze_p53(model):
    model.eval()
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    radii = [2, 5, 10, 15, 20, 30]

    def exp_decay(r, A, alpha, c):
        return A * np.exp(-alpha * r) + c

    with open(P53_DATA_PATH, 'rb') as f:
        p53_data = pickle.load(f)

    print("\n=== ESM-2 充分训练后 p53 原始 L_r 值 ===")
    results = []
    for sample in p53_data:
        local_seq = sample['local_seq_full']
        center = sample['center_in_local']
        dh_global = sample['delta_h_global']
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
            L_r = np.linalg.norm(dh_local - dh_global) / (np.linalg.norm(dh_global) + 1e-12)
            L_r_vals.append(L_r)

        print(f"{sample['name']}: L_r = {[f'{x:.3f}' for x in L_r_vals]}")

        # 拟合指数衰减
        try:
            popt, _ = curve_fit(exp_decay, radii, L_r_vals, p0=[3, 0.1, 1], maxfev=10000)
            c_fit = popt[2]
            alpha_fit = popt[1]
        except:
            c_fit = np.nan
            alpha_fit = np.nan

        results.append({
            'mutation': sample['name'],
            'c': c_fit,
            'alpha': alpha_fit,
            'L_r_min': min(L_r_vals),
            'L_r_vals': L_r_vals,
            'NOC': None
        })

    noc_map = {'R175H': 42, 'G245S': 32, 'R249S': 30, 'R282W': 36, 'Y220C': 32}
    for r in results:
        r['NOC'] = noc_map.get(r['mutation'], np.nan)

    df = pd.DataFrame(results).dropna()
    rho, p = spearmanr(df['c'], df['NOC'])
    print(f"\nESM-2 (充分训练): c vs NOC Spearman ρ = {rho:.3f}, p = {p:.4f}")
    print(df[['mutation', 'c', 'alpha', 'L_r_min', 'NOC']])

    # 简易图
    plt.figure(figsize=(6, 5))
    plt.scatter(df['c'], df['NOC'])
    for _, row in df.iterrows():
        plt.annotate(row['mutation'], (row['c'], row['NOC']))
    plt.xlabel('c (irreducible residual)')
    plt.ylabel('NOC clusters')
    plt.title(f'ESM-2 (retrained): c vs NOC (ρ = {rho:.3f})')
    plt.grid(True, alpha=0.3)
    os.makedirs('processed_data/esm2', exist_ok=True)
    plt.savefig('processed_data/esm2/esm2_retrained_c_vs_noc.png', dpi=150)
    plt.show()
    print("图表已保存。")

# ================== 主程序 ==================
if __name__ == "__main__":
    print("开始充分训练 ESM-2 局部编码器...")
    trained_model = train()
    print("\n开始分析 p53 突变...")
    analyze_p53(trained_model)