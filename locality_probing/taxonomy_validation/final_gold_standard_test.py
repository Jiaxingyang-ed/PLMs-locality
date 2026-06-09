import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import os

# ================== 加载数据 ==================
df_100 = pd.read_csv('processed_data/scale_validation/100_mutations_convergence_params.csv')

# ================== 金标准分类 ==================
# Group A (局部依赖)
group_A_muts = [
    # p53突变使用留出p53后计算的c_Lr值
    ('p53', 'G245S', 'G', 245, 'S', 4.328),
    ('p53', 'R249S', 'R', 249, 'S', 3.474),
    # 需要从100个数据中查找的非p53突变
    ('Factor VIIa', 'T58A', 'T', 58, 'A', None),
    ('Factor VIIa', 'T64A', 'T', 64, 'A', None),
    ('Human Angiotensin-converting enzyme 2', 'E150K', 'E', 150, 'K', None),
    ('Human Angiotensin-converting enzyme 2', 'E160S', 'E', 160, 'S', None),
    ('Chemotaxis protein CheY', 'A97V', 'A', 97, 'V', None),
    ('Chemotaxis protein CheY', 'A101W', 'A', 101, 'W', None),
    ('Chemotaxis protein CheY', 'A113V', 'A', 113, 'V', None),
]

# Group B (全局依赖)
group_B_muts = [
    ('p53', 'R175H', 'R', 175, 'H', 5.511),
    ('p53', 'R282W', 'R', 282, 'W', 3.768),
    ('p53', 'Y220C', 'Y', 220, 'C', 4.201),
    ('Human Angiotensin-converting enzyme 2', 'E22R', 'E', 22, 'R', None),
    ('Human Angiotensin-converting enzyme 2', 'E35S', 'E', 35, 'S', None),
    ('MCP-1', 'L28K', 'L', 28, 'K', None),
    ('Subtilisin BPN', 'I41R', 'I', 41, 'R', None),
    ('IgG1 lambda fab', 'C115L', 'C', 115, 'L', None),
]

# ================== 提取c_Lr ==================
def get_c_Lr_from_100(protein, wt, pos_1idx, mut):
    pos_0idx = pos_1idx - 1
    mask = (df_100['protein'].str.contains(protein, case=False, na=False)) & \
           (df_100['position'] == pos_0idx) & \
           (df_100['wt_aa'] == wt) & \
           (df_100['mut_aa'] == mut)
    matches = df_100[mask]
    if len(matches) > 0:
        return matches.iloc[0]['c_Lr']
    return np.nan

vals_A, vals_B = [], []

print("=== Group A ===")
for item in group_A_muts:
    protein, mut_name, wt, pos, mut, hardcoded = item
    c = hardcoded if hardcoded is not None else get_c_Lr_from_100(protein, wt, pos, mut)
    if not np.isnan(c):
        vals_A.append(c)
        print(f"  {protein} {mut_name}: c_Lr = {c:.3f}")
    else:
        print(f"  ⚠️ 未找到: {protein} {mut_name}")

print("\n=== Group B ===")
for item in group_B_muts:
    protein, mut_name, wt, pos, mut, hardcoded = item
    c = hardcoded if hardcoded is not None else get_c_Lr_from_100(protein, wt, pos, mut)
    if not np.isnan(c):
        vals_B.append(c)
        print(f"  {protein} {mut_name}: c_Lr = {c:.3f}")
    else:
        print(f"  ⚠️ 未找到: {protein} {mut_name}")

print(f"\nGroup A: n={len(vals_A)}, mean c_Lr = {np.mean(vals_A):.3f}")
print(f"Group B: n={len(vals_B)}, mean c_Lr = {np.mean(vals_B):.3f}")

if len(vals_A) >= 3 and len(vals_B) >= 3:
    stat, p = mannwhitneyu(vals_B, vals_A, alternative='greater')
    print(f"Mann-Whitney p = {p:.5f}")
else:
    print("样本量不足")

# 箱线图
plt.figure(figsize=(6,5))
plt.boxplot([vals_A, vals_B], labels=['Group A (local)', 'Group B (global)'], patch_artist=True)
plt.ylabel('c_Lr')
plt.title('Gold Standard Validation')
plt.grid(True, alpha=0.3)
os.makedirs('locality_probing/taxonomy_validation', exist_ok=True)
plt.savefig('locality_probing/taxonomy_validation/final_gold_standard.png', dpi=150)
plt.show()