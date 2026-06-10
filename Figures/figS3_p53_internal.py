import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
})

# p53 内部数据
group_A = [3.474, 4.328]   # R249S, G245S
group_B = [5.511, 3.768, 4.201]  # R175H, R282W, Y220C
labels_A = ["R249S", "G245S"]
labels_B = ["R175H", "R282W", "Y220C"]

fig, ax = plt.subplots(figsize=(6, 5))

# 箱线图
bp = ax.boxplot([group_A, group_B], patch_artist=True,
                labels=["Group A\n(Local mutations)", "Group B\n(Global mutations)"])
bp['boxes'][0].set_facecolor("#1F77B4")
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor("#D62728")
bp['boxes'][1].set_alpha(0.6)

# 叠加散点并标注
colors = {"R175H": "#D62728", "R282W": "#FF7F0E", "Y220C": "#9467BD",
          "G245S": "#1F77B4", "R249S": "#2CA02C"}
all_vals = group_A + group_B
all_labels = labels_A + labels_B
all_colors = [colors[n] for n in all_labels]

for i, (val, name, color) in enumerate(zip(all_vals, all_labels, all_colors)):
    x = 1 if i < 2 else 2
    x_jitter = x + np.random.normal(0, 0.05)
    ax.scatter(x_jitter, val, color=color, s=80, edgecolors="white", linewidths=1.0, zorder=5)
    ax.annotate(name, (x_jitter, val), textcoords="offset points",
                xytext=(8, 4), fontsize=9, color=color, weight="bold")

# 标注均值差异
mean_A = np.mean(group_A)
mean_B = np.mean(group_B)
ax.text(0.98, 0.95, f"Mean Group A: {mean_A:.2f}\nMean Group B: {mean_B:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_ylabel("Irreducible Residual Plateau $c$")
ax.set_title("p53 Internal Gold Standard Validation", fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.3)

plt.tight_layout()
plt.savefig("Figures/figS3_p53_internal.pdf")
print("figS3_p53_internal.pdf saved.")
