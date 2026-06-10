import torch, pickle, numpy as np, os, sys
from scipy.stats import mannwhitneyu
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESIDUE_EMB_DIR
from train_local_encoder import LocalSeqEncoder, MAX_WINDOW_LEN, DEVICE

MODEL_PATH = "locality_probing/local_encoder/local_encoder.pt"
model = LocalSeqEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 加载突变数据
with open(os.path.join(RESIDUE_EMB_DIR, 'mutation_deltas.pkl'), 'rb') as f:
    data = pickle.load(f)

aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

def compute_L(sample):
    seq_idx = [aa_to_idx.get(aa, 0) for aa in sample['local_seq']]
    if len(seq_idx) < MAX_WINDOW_LEN:
        seq_idx += [0] * (MAX_WINDOW_LEN - len(seq_idx))
    else:
        seq_idx = seq_idx[:MAX_WINDOW_LEN]
    seq_tensor = torch.tensor(seq_idx).unsqueeze(0).to(DEVICE)
    mut_tensor = torch.tensor([aa_to_idx.get(sample['mut_aa'], 0)]).to(DEVICE)
    delta_h_global = sample['delta_h_global']
    with torch.no_grad():
        delta_h_local = model(seq_tensor, mut_tensor).squeeze().cpu().numpy()
    norm_l = np.linalg.norm(delta_h_local)
    norm_g = np.linalg.norm(delta_h_global)
    if norm_l < 1e-6 or norm_g < 1e-6:
        return 0.0
    cos = np.dot(delta_h_local, delta_h_global) / (norm_l * norm_g)
    return 1.0 - cos

# 定义金标准突变
gold_conform = [
    ('p53 core domain', 175, 'R', 'H'),
    ('p53 core domain', 282, 'R', 'W'),
    ('p53 core domain', 245, 'G', 'S'),
    ('p53_wt', 175, 'R', 'H'),
    ('p53_wt', 282, 'R', 'W'),
    ('p53_wt', 245, 'G', 'S'),
]
gold_conserv = [
    ('Subtilisin Carlsberg', 156, 'G', 'A'),
    ('Subtilisin BPN', 156, 'G', 'A'),
    ('Bovine trypsin', 60, 'S', 'T'),
    ('Bovine trypsin', 100, 'K', 'R'),
]

group_conf = []
group_cons = []
for d in data:
    prot, pos, wt, mut = d['protein'], d['position'], d['wt_aa'], d['mut_aa']
    for g in gold_conform:
        if g[0] in prot and pos == g[1] and wt == g[2] and mut == g[3]:
            group_conf.append(compute_L(d))
            break
    for g in gold_conserv:
        if g[0] in prot and pos == g[1] and wt == g[2] and mut == g[3]:
            group_cons.append(compute_L(d))
            break

print(f"Found {len(group_conf)} conformational, {len(group_cons)} conservative.")
if len(group_conf) > 0 and len(group_cons) > 0:
    stat, p = mannwhitneyu(group_conf, group_cons, alternative='greater')
    print(f"Conformational L mean: {np.mean(group_conf):.4f}")
    print(f"Conservative L mean: {np.mean(group_cons):.4f}")
    print(f"Mann-Whitney p = {p:.5f}")
else:
    print("Not enough gold-standard mutations found. Need to supplement.")