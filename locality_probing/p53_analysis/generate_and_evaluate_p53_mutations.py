#!/usr/bin/env python3
"""
为p53五个文献验证的突变生成Δh数据，并用多维度L度量评估局部性偏差。
"""

import torch, pickle, numpy as np, os, sys, pandas as pd
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

# ================== 路径与配置 ==================
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR, RAW_P53_FASTA

DEVICE = "cpu"
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
LOCAL_ENCODER_PATH = "locality_probing/local_encoder/local_encoder.pt"

# ================== 加载p53序列 ==================
from Bio import SeqIO
with open(RAW_P53_FASTA) as f:
    for record in SeqIO.parse(f, "fasta"):
        p53_seq = str(record.seq)
        break
print(f"p53 sequence length: {len(p53_seq)}")

# ================== 定义五个突变 ==================
# (名称, 1-indexed位置, wt, mut, 预期类别)
p53_mutations = [
    ("R175H", 175, "R", "H", "global_disruptive"),
    ("G245S", 245, "G", "S", "local"),
    ("R249S", 249, "R", "S", "local"),
    ("R282W", 282, "R", "W", "global_disruptive"),
    ("Y220C", 220, "Y", "C", "global_disruptive"),
]

# ================== 第一步：生成Δh数据 ==================
print("Loading ProtT5 for Δh generation...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
prot5_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
prot5_model.eval()

def embed_sequence(seq, max_len=1024):
    """返回每残基的ProtT5嵌入 (L, 1024)"""
    seq = seq[:max_len]
    seq_spaced = " ".join(list(seq))
    inputs = tokenizer(seq_spaced, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = prot5_model(**inputs)
    return outputs.last_hidden_state[0, 1:-1, :].cpu()  # (L, 1024)

print("Generating wild-type p53 embedding...")
wt_emb = embed_sequence(p53_seq)  # (L, 1024)

mutation_data = []
for name, pos_1idx, wt_aa, mut_aa, category in tqdm(p53_mutations, desc="Generating mutations"):
    pos = pos_1idx - 1  # 0-indexed
    if pos < 0 or pos >= len(p53_seq) or p53_seq[pos] != wt_aa:
        print(f"  Warning: {name} position mismatch, expected {wt_aa} at {pos_1idx}, found {p53_seq[pos]}")
        continue
    # 突变序列
    mut_seq = p53_seq[:pos] + mut_aa + p53_seq[pos+1:]
    # 全局突变嵌入
    mut_emb = embed_sequence(mut_seq)
    if pos >= wt_emb.shape[0] or pos >= mut_emb.shape[0]:
        print(f"  Warning: {name} position out of embedding bounds")
        continue
    # 全局Δh
    delta_h_global = (mut_emb[pos] - wt_emb[pos]).numpy()
    # 局部序列窗口（最大r=20对应窗口长度41）
    max_radius = 20
    start = max(0, pos - max_radius)
    end = min(len(p53_seq), pos + max_radius + 1)
    local_seq_full = p53_seq[start:end]
    
    mutation_data.append({
        "name": name,
        "category": category,
        "position": pos,
        "wt_aa": wt_aa,
        "mut_aa": mut_aa,
        "local_seq_full": local_seq_full,
        "delta_h_global": delta_h_global,
        "center_in_local": pos - start  # 突变残基在local_seq_full中的索引
    })
    print(f"  {name}: Δh_global norm = {np.linalg.norm(delta_h_global):.4f}")

# 保存Δh数据
os.makedirs("processed_data/p53_mutations", exist_ok=True)
with open("processed_data/p53_mutations/p53_mutation_deltas.pkl", "wb") as f:
    pickle.dump(mutation_data, f)
print(f"\nSaved {len(mutation_data)} mutation Δh samples.")

# ================== 第二步：多维度L度量 ==================
print("\nLoading Local Encoder...")
MAX_WINDOW_LEN = 41  # 支持最大r=20

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

local_encoder = LocalSeqEncoder().to(DEVICE)
local_encoder.load_state_dict(torch.load(LOCAL_ENCODER_PATH, map_location=DEVICE))
local_encoder.eval()

aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

def predict_delta_h_local(local_seq, center_pos, radius):
    """提取以center_pos为中心、半径radius的局部窗口，用编码器预测Δh"""
    start = max(0, center_pos - radius)
    end = min(len(local_seq), center_pos + radius + 1)
    sub_seq = local_seq[start:end]
    # 编码为索引
    seq_idx = [aa_to_idx.get(aa, 0) for aa in sub_seq]
    # 填充到MAX_WINDOW_LEN
    if len(seq_idx) < MAX_WINDOW_LEN:
        seq_idx += [0] * (MAX_WINDOW_LEN - len(seq_idx))
    else:
        seq_idx = seq_idx[:MAX_WINDOW_LEN]
    seq_tensor = torch.tensor(seq_idx).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        delta_h_local = local_encoder(seq_tensor, None).squeeze().cpu().numpy()
    return delta_h_local

def compute_L_measures(delta_h_local, delta_h_global):
    """计算三个维度的L值"""
    delta_h_local = np.asarray(delta_h_local)
    delta_h_global = np.asarray(delta_h_global)
    norm_l = np.linalg.norm(delta_h_local)
    norm_g = np.linalg.norm(delta_h_global)
    if norm_g < 1e-12:
        return 0.0, 0.0, 0.0
    # L_θ: 方向偏差
    if norm_l < 1e-12:
        L_theta = 1.0  # 局部预测为零向量，最大偏差
    else:
        cos = np.dot(delta_h_local, delta_h_global) / (norm_l * norm_g)
        cos = np.clip(cos, -1.0, 1.0)
        L_theta = 1.0 - cos
    # L_m: log-模长比
    eps = 1e-8
    L_m = np.abs(np.log((norm_l + eps) / (norm_g + eps)))
    # L_r: 相对残差
    L_r = np.linalg.norm(delta_h_local - delta_h_global) / (norm_g + eps)
    return L_theta, L_m, L_r

# 计算多窗口L值
radii = [5, 10, 20]
results = []

for sample in mutation_data:
    local_seq = sample["local_seq_full"]
    center = sample["center_in_local"]
    delta_h_global = sample["delta_h_global"]
    
    for r in radii:
        delta_h_local = predict_delta_h_local(local_seq, center, r)
        L_theta, L_m, L_r = compute_L_measures(delta_h_local, delta_h_global)
        results.append({
            "mutation": sample["name"],
            "category": sample["category"],
            "radius": r,
            "L_theta": L_theta,
            "L_m": L_m,
            "L_r": L_r
        })
        print(f"{sample['name']} r={r}: L_θ={L_theta:.4f}, L_m={L_m:.4f}, L_r={L_r:.4f}")

# 保存结果
df = pd.DataFrame(results)
df.to_csv("processed_data/p53_mutations/p53_locality_measures.csv", index=False)
print("\nAll L measures saved to processed_data/p53_mutations/p53_locality_measures.csv")