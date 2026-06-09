#!/usr/bin/env python3
import torch, numpy as np, pandas as pd, os, sys
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, EsmModel

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import RESULT_DIR

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
DEVICE = "cpu"
WINDOW_RADIUS = 10

print("Loading ESM-2 35M...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def get_residue_embedding(sequence, position, window_radius=None):
    if window_radius is not None:
        start = max(0, position - window_radius)
        end = min(len(sequence), position + window_radius + 1)
        subseq = sequence[start:end]
        subpos = position - start
        inputs = tokenizer(subseq, return_tensors="pt").to(DEVICE)
    else:
        subseq = sequence
        subpos = position
        inputs = tokenizer(subseq, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]
    return hidden[subpos + 1].cpu().numpy()

def mutate(seq, pos, new_aa):
    return seq[:pos] + new_aa + seq[pos+1:]

def compute_locality_deviation(seq, pos, wt_aa, mut_aa, radius=WINDOW_RADIUS):
    mut_seq = mutate(seq, pos, mut_aa)
    wt_global = get_residue_embedding(seq, pos, window_radius=None)
    mut_global = get_residue_embedding(mut_seq, pos, window_radius=None)
    delta_global = mut_global - wt_global
    wt_local = get_residue_embedding(seq, pos, window_radius=radius)
    mut_local = get_residue_embedding(mut_seq, pos, window_radius=radius)
    delta_local = mut_local - wt_local
    norm_g = np.linalg.norm(delta_global)
    norm_l = np.linalg.norm(delta_local)
    if norm_g < 1e-12 or norm_l < 1e-12:
        return 0.0
    cos = np.dot(delta_local, delta_global) / (norm_l * norm_g)
    return 1.0 - cos

p53_seq = ("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
           "DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK"
           "SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHH"
           "ERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCN"
           "SSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHEL"
           "PPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEP"
           "GGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD")

mutations = [
    ("L201I", 200, 'L', 'I', 'A'),
    ("D281E", 280, 'D', 'E', 'A'),
    ("A276S", 275, 'A', 'S', 'A'),
    ("V272I", 271, 'V', 'I', 'A'),
    ("R175H", 174, 'R', 'H', 'B'),
    ("R248Q", 247, 'R', 'Q', 'B'),
    ("R273H", 272, 'R', 'H', 'B'),
    ("R282W", 281, 'R', 'W', 'B'),
    ("G245S", 244, 'G', 'S', 'B'),
    ("Y220C", 219, 'Y', 'C', 'B'),
]

results = []
for name, pos, wt, mut, grp in mutations:
    fail = compute_locality_deviation(p53_seq, pos, wt, mut)
    results.append({"name": name, "group": grp, "failure": fail})
    print(f"  {name} ({grp}) -> failure = {fail:.4f}")

df = pd.DataFrame(results)
scores_A = df[df["group"] == "A"]["failure"]
scores_B = df[df["group"] == "B"]["failure"]
stat, p = mannwhitneyu(scores_B, scores_A, alternative='greater')
print(f"\nMann-Whitney U test (B > A): statistic = {stat:.3f}, p = {p:.5f}")

plt.figure(figsize=(6, 5))
plt.boxplot([scores_A, scores_B], labels=["Conservative (A)", "Disruptive (B)"])
plt.ylabel("Locality Deviation (1 - cos)")
plt.title("Locality Deviation by Mutation Type")
os.makedirs(RESULT_DIR, exist_ok=True)
plt.savefig(os.path.join(RESULT_DIR, "task5_concept_validation.png"))
print(f"Plot saved to {RESULT_DIR}/task5_concept_validation.png")
