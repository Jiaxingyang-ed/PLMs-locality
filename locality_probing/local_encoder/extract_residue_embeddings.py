
import torch
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from Bio import SeqIO
import pandas as pd
import os, sys, requests, time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RAW_P53_FASTA, SKEMPI_CLEAN, RESIDUE_EMB_DIR

print("Script started.")

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
DEVICE = "cpu"
MAX_LENGTH = 1024

# 收集序列
print("Collecting protein sequences...")
sequences = {}

# p53
with open(RAW_P53_FASTA) as f:
    for record in SeqIO.parse(f, "fasta"):
        sequences["p53_wt"] = str(record.seq)

# SKEMPI proteins
def fetch_uniprot_sequence(protein_name):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=fasta"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and ">" in r.text:
            seq = "".join(r.text.split("\n")[1:])
            return seq
    except:
        pass
    return None

df_skempi = pd.read_csv(SKEMPI_CLEAN)
protein_names = df_skempi["mutated_protein"].dropna().unique()
print(f"Found {len(protein_names)} unique proteins in SKEMPI.")

for name in protein_names:
    if name in sequences:
        continue
    clean = name.strip()
    seq = fetch_uniprot_sequence(clean)
    if seq:
        sequences[clean] = seq
        print(f"  {clean}: {len(seq)} aa")
    else:
        print(f"  FAILED: {clean}")
    time.sleep(0.2)

print(f"Total sequences: {len(sequences)}")

# 加载模型
print("Loading ProtT5-XL...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# 提取嵌入
os.makedirs(RESIDUE_EMB_DIR, exist_ok=True)
for name, seq in sequences.items():
    if len(seq) > MAX_LENGTH:
        seq = seq[:MAX_LENGTH]
    seq_spaced = " ".join(list(seq))
    inputs = tokenizer(seq_spaced, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.last_hidden_state[0, 1:-1, :].half()
    save_path = os.path.join(RESIDUE_EMB_DIR, f"{name}.pt")
    torch.save(hidden, save_path)
    print(f"Saved {name} -> {save_path} ({hidden.shape})")

print("All done.")
