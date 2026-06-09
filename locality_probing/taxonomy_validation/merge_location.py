#!/usr/bin/env python3
"""从SKEMPI CSV中提取结构注释并合并到规模验证数据中"""

import pandas as pd, pickle, os, sys
sys.path.append('scripts')
from config import RESIDUE_EMB_DIR, SKEMPI_CLEAN

# 1. 加载SKEMPI原始数据（包含iMutation_Location(s)）
skempi = pd.read_csv(SKEMPI_CLEAN)
# 确保有位置和蛋白名列（列名可能是mutated_protein, position等）
print("SKEMPI columns:", list(skempi.columns))

# 2. 加载规模验证数据
df = pd.read_csv('processed_data/scale_validation/100_mutations_convergence_params.csv')

# 3. 合并：基于 (protein, position, mut_aa) 匹配
# 注意：SKEMPI中protein名和位置格式可能需要对齐
# 这里假设SKEMPI有 'mutated_protein', 'position', 'mut_aa' 列
if 'iMutation_Location(s)' in skempi.columns:
    loc_col = 'iMutation_Location(s)'
elif 'iMutation_Location' in skempi.columns:
    loc_col = 'iMutation_Location'
else:
    print("SKEMPI CSV中未找到结构注释列，请检查列名")
    exit()

# 构建映射
loc_map = {}
for _, row in skempi.iterrows():
    key = (str(row['mutated_protein']), int(row['position'])-1, row['mut_aa'])  # 0-indexed
    loc_map[key] = row[loc_col]

# 应用到df
df['location'] = df.apply(lambda r: loc_map.get((r['protein'], r['position'], r['mut_aa']), None), axis=1)
print(f"成功匹配结构注释的突变数: {df['location'].notna().sum()}")

# 保存
df.to_csv('processed_data/scale_validation/100_mutations_with_location.csv', index=False)
print("保存完成。")