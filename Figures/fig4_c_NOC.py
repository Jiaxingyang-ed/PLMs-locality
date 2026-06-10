import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
})

radii = np.array([2, 5, 10, 15, 20, 30])
L_r = {
    "R175H":  [17.99, 14.64, 10.31, 7.47, 5.25, 5.25],
    "G245S":  [15.10, 12.40, 8.87,  5.93, 4.34, 4.34],
    "R249S":  [11.97, 9.77,  6.90,  4.82, 3.47, 3.47],
    "R282W":  [12.49, 10.38, 7.55,  4.83, 3.47, 3.47],
    "Y220C":  [14.81, 12.37, 8.33,  5.70, 3.94, 3.94],
}
c_vals = [4.015, 3.174, 2.623, 2.326, 2.739]
noc_vals = [42, 32, 30, 36, 32]
mut_names = ["R175H", "G245S", "R249S", "R282W", "Y220C"]
colors = {"R175H": "#D62728", "G245S": "#1F77B4", "R249S": "#2CA02C",
          "R282W": "#FF7F0E", "Y220C": "#9467BD"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ===== 右面板：c vs NOC =====
rho, p = spearmanr(c_vals, noc_vals)

# 趋势线和置信带
z = np.polyfit(c_vals, noc_vals, 1)
p_line = np.poly1d(z)
c_range = np.linspace(min(c_vals)*0.9, max(c_vals)*1.1, 50)
pred = p_line(c_range)
ax2.fill_between(c_range, pred-2, pred+2, color="gray", alpha=0.1)
ax2.plot(c_range, pred, color="gray", linestyle="--", linewidth=1.8)

# 散点
for name, cx, ny in zip(mut_names, c_vals, noc_vals):
    ax2.scatter(cx, ny, color=colors[name], s=180, edgecolors="white", linewidths=1.0, zorder=5)
    # 调整标注位置
    dx, dy = 0.15, 2
    if name == "R175H": dy = -3
    if name == "Y220C": dx = -0.5
    ax2.annotate(name, (cx, ny), textcoords="offset points", xytext=(dx*10, dy*10),
                 fontsize=10, color=colors[name], weight="bold")

# 统计量放在图内
ax2.text(0.95, 0.05, f"Spearman $\\rho$ = {rho:.2f}\n$p$ = {p:.2f}",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax2.set_xlabel("Irreducible Residual Plateau $c$")
ax2.set_ylabel("Conformational Clusters (NOC)")
ax2.set_title("$c$ Correlates with Conformational Heterogeneity", fontweight="bold")
ax2.grid(axis="y", linestyle=":", alpha=0.3)

plt.tight_layout()
plt.savefig("Figures/fig3_Lr_NOC.pdf")
print("fig3_Lr_NOC.pdf saved.")
