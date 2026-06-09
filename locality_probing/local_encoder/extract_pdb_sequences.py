#!/usr/bin/env python3
"""Extract sequences from downloaded PDB files and match with SKEMPI mutations."""

import os, pickle, glob
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from config import SKEMPI_CLEAN, RESIDUE_EMB_DIR

# 1. Find all PDB files in legacy
pdb_dir = "legacy/data/original data/mutation_data"
pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
print(f"Found {len(pdb_files)} PDB files")

# 2. Extract sequences from PDB files
parser = PDBParser(QUIET=True)
pdb_sequences = {}  # pdb_id_chain -> sequence
for pdb_file in tqdm(pdb_files):
    try:
        structure = parser.get_structure('temp', pdb_file)
        for model in structure:
            for chain in model:
                residues = [res for res in chain.get_residues() if res.get_id()[0]==' ']
                if len(residues) < 5:
                    continue
                three_codes = [res.get_resname() for res in residues]
                seq = seq1("".join(three_codes))
                # Use filename + chain as key
                key = os.path.basename(pdb_file).replace(".pdb","") + "_" + chain.id
                pdb_sequences[key] = seq
    except:
        pass

print(f"Extracted {len(pdb_sequences)} sequences from PDB files")

# 3. Load SKEMPI mutations
df = pd.read_csv(SKEMPI_CLEAN)

# 4. Match mutations to PDB sequences
# SKEMPI entries have a column '#Pdb' like '1CSE_E_I', which we can parse
matched = []
for idx, row in df.iterrows():
    pdb_entry = str(row.get('#Pdb', ''))
    if not pdb_entry or pd.isna(pdb_entry):
        continue
    parts = pdb_entry.split('_')
    if len(parts) < 2:
        continue
    pdb_id = parts[0].upper()
    chain = parts[1]
    # Search for matching PDB file
    for key, seq in pdb_sequences.items():
        if pdb_id in key.upper() and chain in key.upper():
            matched.append({
                'pdb_entry': pdb_entry,
                'pdb_key': key,
                'sequence': seq,
                'position': int(row['position']),
                'wt_aa': row['wt_aa'],
                'mut_aa': row['mut_aa'],
                'ddG': row.get('ddG', 0.0),
                'mutated_protein': row['mutated_protein']
            })
            break

print(f"Matched {len(matched)} mutations to PDB sequences")

# 5. Save mapping
output_path = os.path.join(RESIDUE_EMB_DIR, 'pdb_matched_mutations.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(matched, f)
print(f"Saved to {output_path}")