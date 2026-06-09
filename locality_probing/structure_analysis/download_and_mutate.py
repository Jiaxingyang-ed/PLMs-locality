#!/usr/bin/env python3
"""Download PDB structures and create rough mutant models for RMSD/contact analysis."""

import os, requests
from Bio.PDB import PDBParser, PDBIO

# 蛋白到 PDB ID 和链的映射（基于常识）
TARGET_MAP = {
    'Bovine alpha-chymotrypsin': ('1acb', 'E'),
    'Subtilisin BPN': ('1sbn', 'E'),
    'Subtilisin Carlsberg': ('1cse', 'E'),
    'Human leukocyte elastase': ('1hne', 'A'),
    'HyHEL-10': ('1c08', 'H'),         # 重链
    'Bovine trypsin': ('1bt0', 'A'),
    'Factor VIIa': ('1dan', 'H'),
    'Chemotaxis protein CheY': ('1chy', 'A'),
    'Streptomyces griseus proteinase B': ('1sgc', 'E'),
}

# 金标准突变列表 (蛋白名, 位置1-indexed, wt, mut, 原始分类)
MUTATIONS = [
    ('Bovine alpha-chymotrypsin', 9, 'I', 'D', 'disruptive'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'E', 'disruptive'),
    ('Subtilisin BPN', 41, 'I', 'P', 'disruptive'),
    ('Subtilisin Carlsberg', 41, 'I', 'P', 'disruptive'),
    ('Human leukocyte elastase', 30, 'I', 'R', 'disruptive'),
    ('Subtilisin BPN', 41, 'I', 'G', 'disruptive'),
    ('HyHEL-10', 34, 'L', 'A', 'disruptive'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'P', 'disruptive'),
    ('Human leukocyte elastase', 30, 'I', 'M', 'disruptive'),
    ('Bovine trypsin', 5, 'I', 'A', 'disruptive'),
    ('Subtilisin BPN', 41, 'I', 'M', 'conservative'),
    ('Factor VIIa', 56, 'T', 'A', 'conservative'),
    ('Chemotaxis protein CheY', 88, 'A', 'V', 'conservative'),
    ('Bovine alpha-chymotrypsin', 9, 'I', 'L', 'conservative'),
    # HyHEL-10 H34A 未在数据中找到，跳过
    ('Bovine alpha-chymotrypsin', 9, 'I', 'G', 'conservative'),
    ('Streptomyces griseus proteinase B', 3, 'I', 'A', 'conservative'),
    # Factor VIIa T16A 未在数据中找到，跳过
    ('Subtilisin BPN', 41, 'I', 'Y', 'conservative'),
    ('Chemotaxis protein CheY', 48, 'A', 'C', 'conservative'),
]

def download_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local = f"{pdb_id}.pdb"
    if not os.path.exists(local):
        print(f"下载 {pdb_id}...")
        r = requests.get(url)
        with open(local, 'w') as f:
            f.write(r.text)
    return local

def create_mutant(pdb_file, chain, position, mut_aa):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('wt', pdb_file)
    # 在指定位点替换残基名
    for model in structure:
        for c in model:
            if c.id == chain:
                for res in c:
                    if res.get_id()[1] == position:
                        res.resname = mut_aa
                        print(f"  突变 {position} 为 {mut_aa}")
    # 保存
    mutant_file = f"{pdb_file.replace('.pdb','')}_{chain}_{position}{mut_aa}.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(mutant_file)
    return mutant_file

if __name__ == "__main__":
    for mut in MUTATIONS:
        protein, pos, wt, mt, group = mut
        if protein not in TARGET_MAP:
            print(f"未映射 PDB: {protein}")
            continue
        pdb_id, chain = TARGET_MAP[protein]
        pdb_file = download_pdb(pdb_id)
        print(f"处理 {protein} {wt}{pos}{mt}")
        try:
            create_mutant(pdb_file, chain, pos, mt)
        except Exception as e:
            print(f"  失败: {e}")