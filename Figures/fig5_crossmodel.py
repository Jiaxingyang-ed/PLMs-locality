import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
})

radii = np.array([2, 5, 10, 15, 20, 30])

# ===== ProtT5 数据 =====
L_r_prot5 = {
    "R175H": [17.99, 14.64, 10.31, 7.47, 5.25, 5.25],
    "G245S": [15.10, 12.40, 8.87,  5.93, 4.34, 4.34],
    "R249S": [11.97, 9.77,  6.90,  4.82, 3.47, 3.47],
    "R282W": [12.49, 10.38, 7.55,  4.83, 3.47, 3.47],
    "Y220C": [14.81, 12.37, 8.33,  5.70, 3.94, 3.94],
}

# ===== ESM-2 数据 =====
L_r_esm2 = {
    "R175H": [2.298, 1.789, 1.319, 1.200, 1.431, 2.316],
    "G245S": [2.080, 1.664, 1.193, 1.144, 1.449, 2.202],
    "R249S": [2.577, 1.918, 1.315, 1.245, 1.713, 2.686],
    "R282W": [2.554, 1.908, 1.353, 1.256, 1.630, 2.690],
    "Y220C": [2.460, 1.942, 1.389, 1.305, 1.675, 2.628],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ===== 左面板：ProtT5 (蓝色系) =====
prot5_blue = "#2C5F8A"
for name in L_r_prot5:
    ax1.plot(radii, L_r_prot5[name], color=prot5_blue, linewidth=1.5, alpha=0.6, marker="o", markersize=5)
# 均值线
mean_prot5 = np.mean(list(L_r_prot5.values()), axis=0)
ax1.plot(radii, mean_prot5, color=prot5_blue, linewidth=3.0, label="ProtT5 mean")

ax1.set_ylim(2, 20)
ax1.set_xlim(0, 35)
ax1.set_xlabel("Window Radius (residues)")
ax1.set_ylabel("Relative Residual $L_r$")
ax1.set_title("ProtT5: Monotonic Convergence", fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(axis="y", linestyle=":", alpha=0.3)
ax1.annotate("Plateau $c$", xy=(25, 5.5), fontsize=10, color=prot5_blue,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# ===== 右面板：ESM-2 (红色系) =====
esm2_red = "#A53A3A"
for name in L_r_esm2:
    ax2.plot(radii, L_r_esm2[name], color=esm2_red, linewidth=1.5, alpha=0.6, marker="s", markersize=5)
mean_esm2 = np.mean(list(L_r_esm2.values()), axis=0)
ax2.plot(radii, mean_esm2, color=esm2_red, linewidth=3.0, label="ESM-2 mean")

ax2.set_ylim(1, 3)
ax2.set_xlim(0, 35)
ax2.set_xlabel("Window Radius (residues)")
ax2.set_ylabel("Relative Residual $L_r$")
ax2.set_title("ESM-2: “U” Shaped Rebound", fontweight="bold")
ax2.legend(loc="upper right")
ax2.grid(axis="y", linestyle=":", alpha=0.3)
ax2.annotate("Overfit at\nlarge windows", xy=(28, 2.6), fontsize=10, color=esm2_red,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Figures/fig5_crossmodel.pdf")
print("fig5_crossmodel.pdf saved.")
