import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
})

# 来自 final_gold_standard_test.py 的输出
vals_A = [4.328, 3.474, 8.129, 3.217, 10.571, 16.855, 9.837, 3.883, 3.848]
vals_B = [5.511, 3.768, 4.201, 6.307, 6.539, 5.192, 4.463]

stat, p = mannwhitneyu(vals_B, vals_A, alternative='greater')

fig, ax = plt.subplots(figsize=(6, 5))

bp = ax.boxplot([vals_A, vals_B], patch_artist=True,
                labels=["Group A\n(surface/local)", "Group B\n(core/interface)"])
bp['boxes'][0].set_facecolor("#1F77B4")
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor("#D62728")
bp['boxes'][1].set_alpha(0.6)

# 叠加散点
for i, (vals, color) in enumerate([(vals_A, "#1F77B4"), (vals_B, "#D62728")]):
    x = np.random.normal(i+1, 0.06, size=len(vals))
    ax.scatter(x, vals, color=color, alpha=0.8, edgecolors="white", linewidths=0.8, s=60)

ax.text(0.95, 0.95, f"Mann-Whitney $p$ = {p:.2f}\n($n$ = {len(vals_A)}+{len(vals_B)})",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_ylabel("Irreducible Residual Plateau $c$")
ax.set_title("Cross-protein Gold Standard Validation", fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.3)

plt.tight_layout()
plt.savefig("Figures/figS2_gold_standard.pdf")
print("figS2_gold_standard.pdf saved.")
