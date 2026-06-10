#!/usr/bin/env python3
"""Fig 2 – Optimized: wider y‑range, compact layout, clear non‑convergence."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
})

radii = [2, 5, 10, 15, 20, 30]
L_theta = {
    "R175H":  [1.083, 1.043, 1.166, 1.158, 1.165, 1.159],
    "G245S":  [1.115, 1.145, 1.243, 1.137, 1.138, 1.033],
    "R249S":  [0.946, 1.205, 1.118, 1.094, 1.094, 0.915],
    "R282W":  [0.940, 0.865, 0.954, 0.995, 0.995, 0.890],
    "Y220C":  [1.111, 1.215, 0.782, 0.723, 0.724, 0.764],
}

global_muts = [
    ("R175H", "#E41A1C", "o", "-"),
    ("R282W", "#FF7F00", "D", "-"),
    ("Y220C", "#984EA3", "v", "-"),
]
local_muts = [
    ("G245S", "#377EB8", "s", "--"),
    ("R249S", "#4DAF4A", "^", "--"),
]

# 使用适合论文栏宽的画布尺寸（7×4 英寸）
fig, ax = plt.subplots(figsize=(7, 4))

for name, color, marker, ls in global_muts:
    ax.plot(radii, L_theta[name],
            color=color, marker=marker, markersize=6,
            linewidth=2.0, linestyle=ls, alpha=0.9, label=name)

for name, color, marker, ls in local_muts:
    ax.plot(radii, L_theta[name],
            color=color, marker=marker, markersize=6,
            linewidth=2.0, linestyle=ls, alpha=0.9,
            markerfacecolor="white", markeredgewidth=1.5, label=name)

# 加宽 y 范围，让数据不“顶天”
ax.set_ylim(-0.1, 2.1)

# 理论最大值带（扩大到 0.9–1.1）
ax.axhspan(0.9, 1.1, color="gray", alpha=0.06, zorder=0)
ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(31, 1.02, "max", fontsize=8, color="gray", ha="center", va="bottom")

ax.set_xlabel("Window Radius (residues)")
ax.set_ylabel("Angular Deviation $L_\\theta$")
ax.set_title("Directional Information Is Inherently Global",
             fontweight="bold", pad=10)

ax.grid(axis="y", linestyle=":", alpha=0.4, linewidth=0.6)

# 分组图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0],color="none",marker="",linestyle="",label="Global disruption"),
    Line2D([0],[0],color="#E41A1C",marker="o",linestyle="-",label="R175H"),
    Line2D([0],[0],color="#FF7F00",marker="D",linestyle="-",label="R282W"),
    Line2D([0],[0],color="#984EA3",marker="v",linestyle="-",label="Y220C"),
    Line2D([0],[0],color="none",marker="",linestyle="",label="Local perturbation"),
    Line2D([0],[0],color="#377EB8",marker="s",linestyle="--",
            markerfacecolor="white",markeredgewidth=1.5,label="G245S"),
    Line2D([0],[0],color="#4DAF4A",marker="^",linestyle="--",
            markerfacecolor="white",markeredgewidth=1.5,label="R249S"),
]
ax.legend(handles=legend_elements, ncol=2, loc="upper right",
          frameon=True, fancybox=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("fig2_Ltheta.pdf", format="pdf")
plt.show()
print("fig2_Ltheta.pdf saved.")