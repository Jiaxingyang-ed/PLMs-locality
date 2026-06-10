import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
})

radii = [2, 5, 10, 15, 20, 30]
L_theta = {
    "R175H": [1.083, 1.043, 1.166, 1.158, 1.165, 1.159],
    "G245S": [1.115, 1.145, 1.243, 1.137, 1.138, 1.033],
    "R249S": [0.946, 1.205, 1.118, 1.094, 1.094, 0.915],
    "R282W": [0.940, 0.865, 0.954, 0.995, 0.995, 0.890],
    "Y220C": [1.111, 1.215, 0.782, 0.723, 0.724, 0.764],
}
mean_L = np.mean(list(L_theta.values()), axis=0)

fig, ax = plt.subplots(figsize=(7, 4.5))

# 所有突变线：半透明灰色，区分线型
for name, ls in [("R175H","-"), ("R282W","-"), ("Y220C","-"), ("G245S","--"), ("R249S","--")]:
    ax.plot(radii, L_theta[name], color="gray", linewidth=1.2, linestyle=ls, alpha=0.5)

# 均值线：黑色粗线，视觉焦点
ax.plot(radii, mean_L, color="black", linewidth=2.8, label="Mean")

# y=1 参考线：红色虚线
ax.axhline(y=1.0, color="#D62728", linestyle=":", linewidth=2.0, alpha=0.9)
# 非收敛区域阴影
ax.axhspan(0.92, 1.08, color="#D62728", alpha=0.04)

ax.text(31, 1.02, "max", fontsize=9, color="#D62728", ha="center")
ax.set_ylim(0, 2)
ax.set_xlabel("Window Radius (residues)")
ax.set_ylabel("Angular Deviation $L_{\\theta}$")
ax.set_title("Directional Information Is Irreducibly Global", fontweight="bold", pad=12)

# 结论性标注
ax.text(0.98, 0.08, "No convergence even at $\\pm 30$ residues",
        transform=ax.transAxes, ha="right", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.grid(axis="y", linestyle=":", alpha=0.3)
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("Figures/fig2_Ltheta.pdf")
print("fig2_Ltheta.pdf saved.")
