#!/usr/bin/env python3
"""Calculate RMSD and contact map changes for each mutant pair."""

import numpy as np, os, sys
from Bio.PDB import PDBParser, Superimposer
from scipy.spatial.distance import cdist

def compute_rmsd(wt_file, mut_file, chain):
    parser = PDBParser(QUIET=True)
    try:
        wt_st = parser.get_structure('wt', wt_file)[0][chain]
        mut_st = parser.get_structure('mut', mut_file)[0][chain]
    except:
        return np.nan, np.nan
    wt_ca = [res['CA'] for res in wt_st if 'CA' in res]
    mut_ca = [res['CA'] for res in mut_st if 'CA' in res]
    if len(wt_ca) < 3 or len(mut_ca) < 3:
        return np.nan, np.nan
    # RMSD
    sup = Superimposer()
    sup.set_atoms(wt_ca, mut_ca)
    sup.apply(mut_st.get_atoms())
    rmsd = sup.rms
    # 接触变化
    def get_contacts(struc, cutoff=8.0):
        coords = []
        for res in struc:
            if 'CB' in res:
                coords.append(res['CB'].get_coord())
            elif 'CA' in res:
                coords.append(res['CA'].get_coord())
        coords = np.array(coords)
        if len(coords) < 2:
            return None
        dist = cdist(coords, coords)
        return dist < cutoff
    wt_contacts = get_contacts(wt_st)
    mut_contacts = get_contacts(mut_st)
    if wt_contacts is None or mut_contacts is None:
        return rmsd, np.nan
    changed = np.sum(wt_contacts != mut_contacts) / wt_contacts.size
    return rmsd, changed

# 手动指定文件对（根据前面生成的突变体文件名）
pairs = [
    ('1acb.pdb', '1acb_E_9D.pdb', 'E'),
    ('1acb.pdb', '1acb_E_9E.pdb', 'E'),
    ('1sbn.pdb', '1sbn_E_41P.pdb', 'E'),
    ('1cse.pdb', '1cse_E_41P.pdb', 'E'),
    ('1hne.pdb', '1hne_A_30R.pdb', 'A'),
    ('1sbn.pdb', '1sbn_E_41G.pdb', 'E'),
    ('1c08.pdb', '1c08_H_34A.pdb', 'H'),
    ('1acb.pdb', '1acb_E_9P.pdb', 'E'),
    ('1hne.pdb', '1hne_A_30M.pdb', 'A'),
    ('1bt0.pdb', '1bt0_A_5A.pdb', 'A'),
    ('1sbn.pdb', '1sbn_E_41M.pdb', 'E'),
    ('1dan.pdb', '1dan_H_56A.pdb', 'H'),
    ('1chy.pdb', '1chy_A_88V.pdb', 'A'),
    ('1acb.pdb', '1acb_E_9L.pdb', 'E'),
    ('1acb.pdb', '1acb_E_9G.pdb', 'E'),
    ('1sgc.pdb', '1sgc_E_3A.pdb', 'E'),
    ('1sbn.pdb', '1sbn_E_41Y.pdb', 'E'),
    ('1chy.pdb', '1chy_A_48C.pdb', 'A'),
]

for wt_file, mut_file, chain in pairs:
    if not os.path.exists(mut_file):
        print(f"缺失 {mut_file}")
        continue
    rmsd, cont_change = compute_rmsd(wt_file, mut_file, chain)
    print(f"{mut_file}: RMSD={rmsd:.3f} Å, contact change={cont_change:.3f}")