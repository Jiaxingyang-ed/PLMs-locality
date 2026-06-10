import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
})

# holdout 编码器的 c 值
c_hold = [5.511, 4.328, 3.474, 3.768, 4.201]
noc = [42, 32, 30, 36, 32]
mut_names = ["R175H", "G245S", "R249S", "R282W", "Y220C"]
colors = {"R175H": "#D62728", "G245S": "#1F77B4", "R249S": "#2CA02C",
          "R282W": "#FF7F0E", "Y220C": "#9467BD"}

rho, p = spearmanr(c_hold, noc)

fig, ax = plt.subplots(figsize=(7, 5))

# 趋势线和置信带
z = np.polyfit(c_hold, noc, 1)
p_line = np.poly1d(z)
c_range = np.linspace(min(c_hold)*0.9, max(c_hold)*1.1, 50)
pred = p_line(c_range)
# 简单置信带
ax.fill_between(c_range, pred-2, pred+2, color="gray", alpha=0.1)
ax.plot(c_range, pred, color="gray", linestyle="--", linewidth=1.8)

# 散点
for name, cx, ny in zip(mut_names, c_hold, noc):
    ax.scatter(cx, ny, color=colors[name], s=200, edgecolors="white", linewidths=1.5, zorder=5)
    dx, dy = 0.15, 2
    if name == "R175H": dy = -3
    if name == "Y220C": dx = -0.6
    ax.annotate(name, (cx, ny), textcoords="offset points", xytext=(dx*10, dy*10),
                 fontsize=10, color=colors[name], weight="bold")

# 统计量
ax.text(0.95, 0.05, f"Spearman $\\rho$ = {rho:.2f}\n$p$ = {p:.2f}\n(Holdout, $n$=5)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_xlabel("Irreducible Residual Plateau $c$")
ax.set_ylabel("Conformational Clusters (NOC)")
ax.set_title("Leave-one-family-out Validation", fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.3)

plt.tight_layout()
plt.savefig("Figures/figS1_holdout.pdf")
print("figS1_holdout.pdf saved.")
