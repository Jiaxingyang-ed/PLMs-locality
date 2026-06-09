#!/usr/bin/env python3
"""Expand training data with additional mutation databases (if available)."""
print("Script started...")
import os, sys, pickle, pandas as pd, numpy as np
from tqdm import tqdm

# ---------- 路径修正 ----------
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR

SEQ_PATH = os.path.join(RESIDUE_EMB_DIR, "sequences.pkl")
EXISTING_DATA_PATH = os.path.join(RESIDUE_EMB_DIR, "mutation_deltas.pkl")
EXPANDED_DATA_PATH = os.path.join(RESIDUE_EMB_DIR, "mutation_deltas_expanded.pkl")

# ---------- 主程序 ----------
def main():
    # 加载现有 SKEMPI 数据
    print("Entering main...")
    if not os.path.exists(EXISTING_DATA_PATH):
        print(f"错误：未找到 {EXISTING_DATA_PATH}，请先运行 generate_deltas.py")
        return
    with open(EXISTING_DATA_PATH, 'rb') as f:
        existing_data = pickle.load(f)
    print(f"现有 SKEMPI 突变数：{len(existing_data)}")

    # 尝试加载额外的数据库文件（如果存在）
    new_mutations = []
    additional_files = [
        ("raw_data/protherm_clean.csv", parse_additional_database),
        ("raw_data/ab_bind_clean.csv", parse_additional_database),
        # 可以继续添加更多数据库
    ]

    for filepath, parser in additional_files:
        if os.path.exists(filepath):
            print(f"发现额外数据文件: {filepath}")
            df = pd.read_csv(filepath)
            new = parser(df, existing_data)
            new_mutations.extend(new)
            print(f"  新增突变: {len(new)}")

    if not new_mutations:
        print("未找到额外数据文件。直接复制现有数据作为扩展集。")
        with open(EXPANDED_DATA_PATH, 'wb') as f:
            pickle.dump(existing_data, f)
        print(f"已保存 {len(existing_data)} 条记录到 {EXPANDED_DATA_PATH}")
        print("后续可以使用此数据集（内容与原始数据集相同），或手动添加更多突变后重新运行。")
        return

    # 如果有新突变，生成嵌入（需要 ProtT5）
    print(f"总计新增 {len(new_mutations)} 条突变，开始生成嵌入...")
    # 调用嵌入生成逻辑（与 generate_deltas_v3.py 相同）
    from transformers import T5EncoderModel, T5Tokenizer
    import torch

    with open(SEQ_PATH, 'rb') as f:
        sequences = pickle.load(f)

    MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
    DEVICE = "cpu"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    def embed_sequence(seq):
        seq = seq[:1024]
        seq_spaced = " ".join(list(seq))
        inputs = tokenizer(seq_spaced, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0, 1:-1, :].cpu()

    expanded = list(existing_data)
    for mut in tqdm(new_mutations):
        prot = mut['protein']
        if prot not in sequences:
            continue
        seq = sequences[prot]
        pos = mut['position']
        wt = mut['wt_aa']
        mt = mut['mut_aa']
        if pos < 0 or pos >= len(seq):
            continue
        if seq[pos] != wt:
            found = False
            for offset in range(-5, 6):
                new_pos = pos + offset
                if 0 <= new_pos < len(seq) and seq[new_pos] == wt:
                    pos = new_pos
                    found = True
                    break
            if not found:
                continue
        mut_seq = seq[:pos] + mt + seq[pos+1:]
        wt_emb = embed_sequence(seq)
        mut_emb = embed_sequence(mut_seq)
        if pos >= wt_emb.shape[0] or pos >= mut_emb.shape[0]:
            continue
        delta_h = (mut_emb[pos] - wt_emb[pos]).numpy()
        radius = 10
        start = max(0, pos - radius)
        end = min(len(seq), pos + radius + 1)
        local_seq = seq[start:end]
        expanded.append({
            'protein': prot,
            'position': pos,
            'wt_aa': wt,
            'mut_aa': mt,
            'local_seq': local_seq,
            'delta_h_global': delta_h,
            'ddG': mut.get('ddG', 0.0)
        })

    with open(EXPANDED_DATA_PATH, 'wb') as f:
        pickle.dump(expanded, f)
    print(f"扩展完成，总突变数：{len(expanded)}")

def parse_additional_database(df, existing_data):
    """解析额外的数据库文件（假设列名为 protein, position, wt_aa, mut_aa, ddG）"""
    existing_keys = set((d['protein'], d['position'], d['mut_aa']) for d in existing_data)
    new = []
    for _, row in df.iterrows():
        key = (row['protein'], int(row['position'])-1, row['mut_aa'])
        if key not in existing_keys:
            new.append({
                'protein': row['protein'],
                'position': int(row['position'])-1,
                'wt_aa': row['wt_aa'],
                'mut_aa': row['mut_aa'],
                'ddG': row.get('ddG', 0.0)
            })
    return new

if __name__ == "__main__":
    main()