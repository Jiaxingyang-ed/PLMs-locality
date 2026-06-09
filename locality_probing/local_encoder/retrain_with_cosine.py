#!/usr/bin/env python3
"""
重新训练局部编码器，引入余弦相似度损失项以校准方向预测。
"""

import torch, pickle, numpy as np, os, sys
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR

# ================== 配置 ==================
MAX_WINDOW_LEN = 21
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
DEVICE = "cpu"
LAMBDA_COS = 0.5          # 余弦损失权重
VALID_PATIENCE = 20       # 早停耐心
MODEL_SAVE_PATH = "locality_probing/local_encoder/local_encoder_cosine.pt"

# ================== 模型定义 ==================
class LocalSeqEncoder(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=128, output_dim=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True
        )
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
            local_seq = d['local_seq']
            seq_idx = [aa_to_idx.get(aa, 0) for aa in local_seq]
            if len(seq_idx) < max_len:
                seq_idx += [0] * (max_len - len(seq_idx))
            else:
                seq_idx = seq_idx[:max_len]
            delta_h = d['delta_h_global']
            self.samples.append((
                torch.tensor(seq_idx, dtype=torch.long),
                torch.tensor(delta_h, dtype=torch.float)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ================== 复合损失函数 ==================
def combined_loss(pred, target, lambda_cos=LAMBDA_COS):
    """MSE + 余弦相似度损失"""
    mse = nn.functional.mse_loss(pred, target)
    # 余弦相似度 (batch-wise)
    cos_sim = nn.functional.cosine_similarity(pred, target, dim=1).mean()
    cos_loss = 1.0 - cos_sim
    return mse + lambda_cos * cos_loss, mse.item(), cos_loss.item()

# ================== 训练函数 ==================
def main():
    # 1. 加载数据
    data_path = os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl')
    if not os.path.exists(data_path):
        print(f"数据文件 {data_path} 不存在！请先运行 generate_deltas.py")
        return
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    print(f"总样本数: {len(all_data)}")

    # 2. 蛋白级划分
    proteins = list(set(d['protein'] for d in all_data))
    train_prots, val_prots = train_test_split(proteins, test_size=0.2, random_state=42)
    train_data = [d for d in all_data if d['protein'] in train_prots]
    val_data = [d for d in all_data if d['protein'] in val_prots]
    print(f"训练蛋白: {len(train_prots)}, 验证蛋白: {len(val_prots)}")
    print(f"训练样本: {len(train_data)}, 验证样本: {len(val_data)}")

    train_ds = DeltaDataset(train_data)
    val_ds = DeltaDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 3. 模型
    model = LocalSeqEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练循环
    best_val_cos = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_total = 0.0
        train_mse = 0.0
        train_cos = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for seq, target in pbar:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            pred = model(seq, None)
            loss, mse, cos = combined_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()
            train_mse += mse
            train_cos += cos
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{mse:.4f}",
                'cos': f"{cos:.4f}"
            })
        n_batches = len(train_loader)
        print(f"Epoch {epoch+1} Train: total={train_total/n_batches:.4f}, "
              f"mse={train_mse/n_batches:.4f}, cos_loss={train_cos/n_batches:.4f}")

        # 验证
        model.eval()
        val_cos_total = 0.0
        val_mse_total = 0.0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(DEVICE), target.to(DEVICE)
                pred = model(seq, None)
                cos_sim = nn.functional.cosine_similarity(pred, target, dim=1)
                val_cos_total += (1.0 - cos_sim).sum().item()
                val_mse_total += nn.functional.mse_loss(pred, target).item() * len(seq)
        avg_val_cos = val_cos_total / len(val_ds)
        avg_val_mse = val_mse_total / len(val_ds)
        print(f"Epoch {epoch+1} Val: cos_loss={avg_val_cos:.4f}, mse={avg_val_mse:.4f}")

        # 早停
        if avg_val_cos < best_val_cos:
            best_val_cos = avg_val_cos
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 模型已保存 (best val cos_loss)")
        else:
            patience_counter += 1
            if patience_counter >= VALID_PATIENCE:
                print("早停触发，训练结束。")
                break

    print(f"训练完成。最佳验证余弦损失: {best_val_cos:.4f}")

if __name__ == "__main__":
    main()