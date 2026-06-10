#!/usr/bin/env python3
"""Group A/B Taxonomy Validation: 检验c_Lr的机制特异性"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import os

# ================== 氨基酸属性 ==================
# 分子量 (Da)
MW = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
}
# BLOSUM62 正分组（≥1 视为保守）
BLOSUM62 = {
    'A': {'A':4,'S':1,'T':1,'V':0,'G':0,'P':-1},
    'C': {'C':9},
    'D': {'D':6,'E':2,'N':1},
    'E': {'E':5,'D':2,'Q':2},
    'F': {'F':6,'Y':3,'W':1,'L':0},
    'G': {'G':6,'S':0,'A':0},
    'H': {'H':8,'Y':2,'N':0},
    'I': {'I':4,'L':2,'V':3,'M':1},
    'K': {'K':5,'R':2,'Q':1},
    'L': {'L':4,'I':2,'V':1,'M':2,'F':0},
    'M': {'M':5,'L':2,'I':1,'V':1},
    'N': {'N':6,'D':1,'S':1},
    'P': {'P':7},
    'Q': {'Q':5,'E':2,'K':1,'R':1},
    'R': {'R':5,'K':2,'Q':1},
    'S': {'S':4,'A':1,'T':1,'N':1},
    'T': {'T':5,'S':1},
    'V': {'V':4,'I':3,'L':1},
    'W': {'W':11,'Y':2,'F':1},
    'Y': {'Y':7,'F':3,'W':2}
}

def classify_mutation(wt, mut):
    """根据先验规则分类为 Group A 或 Group B"""
    if wt not in MW or mut not in MW:
        return None

    # Group B 规则
    # 1. 引入脯氨酸
    if mut == 'P':
        return 'B'
    # 2. 引入甘氨酸
    if mut == 'G':
        return 'B'
    # 3. 电荷反转（正→负，负→正）
    positive = {'K', 'R', 'H'}
    negative = {'D', 'E'}
    if (wt in positive and mut in negative) or (wt in negative and mut in positive):
        return 'B'
    # 4. 破坏半胱氨酸（二硫键）
    if wt == 'C' and mut != 'C':
        return 'B'
    # 5. 侧链体积剧烈变化 (>80 Da)
    if abs(MW[wt] - MW[mut]) > 80:
        return 'B'
    # 6. 芳香族核心替换
    aromatic = {'F', 'W', 'Y'}
    if (wt in aromatic and mut not in aromatic) or (wt not in aromatic and mut in aromatic):
        if abs(MW[wt] - MW[mut]) > 40:
            return 'B'

    # Group A 规则
    # 1. 同一BLOSUM62高分组 (≥2)
    if wt in BLOSUM62 and mut in BLOSUM62[wt]:
        if BLOSUM62[wt][mut] >= 2:
            return 'A'
    # 2. 同电荷组
    if (wt in positive and mut in positive) or (wt in negative and mut in negative):
        return 'A'
    # 3. 侧链体积相似 (<20 Da)
    if abs(MW[wt] - MW[mut]) < 20:
        return 'A'

    # 无法分类
    return None

# ================== 加载数据 ==================
df = pd.read_csv('processed_data/scale_validation/100_mutations_convergence_params.csv')
df = df.dropna(subset=['c_Lr'])

# 分类
df['group'] = df.apply(lambda row: classify_mutation(row['wt_aa'], row['mut_aa']), axis=1)
classified = df[df['group'].notna()].copy()
print(f"分类结果：Group A: {len(classified[classified['group']=='A'])} 个，Group B: {len(classified[classified['group']=='B'])} 个，未分类: {len(df) - len(classified)} 个")

# ================== 统计检验 ==================
group_A = classified[classified['group'] == 'A']['c_Lr']
group_B = classified[classified['group'] == 'B']['c_Lr']

if len(group_A) >= 3 and len(group_B) >= 3:
    stat, p = mannwhitneyu(group_B, group_A, alternative='greater')
    print(f"Group A mean c_Lr: {group_A.mean():.3f}")
    print(f"Group B mean c_Lr: {group_B.mean():.3f}")
    print(f"Mann-Whitney U (B > A): p = {p:.5f}")
else:
    print("样本量不足，无法进行统计检验")

# ================== 可视化 ==================
os.makedirs('locality_probing/taxonomy_validation', exist_ok=True)
plt.figure(figsize=(6,5))
plt.boxplot([group_A.values, group_B.values], labels=['Group A (local)', 'Group B (global)'])
plt.ylabel('c_Lr (irreducible global dependency)')
plt.title(f'Mutation Taxonomy Validation\n(Mann-Whitney p = {p:.4f})' if len(group_A)>=3 else 'Mutation Taxonomy Validation')
plt.grid(True, alpha=0.3)
plt.savefig('locality_probing/taxonomy_validation/group_ab_validation.png', dpi=150)
plt.show()

print("图表已保存。")