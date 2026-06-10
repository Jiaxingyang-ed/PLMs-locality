import matplotlib.pyplot as plt
import numpy as np

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
colors = {"R175H": "#D62728", "G245S": "#1F77B4", "R249S": "#2CA02C",
          "R282W": "#FF7F0E", "Y220C": "#9467BD"}

fig, ax = plt.subplots(figsize=(7, 5))

# 其他突变：半透明灰色
for name in ["G245S", "R249S", "R282W", "Y220C"]:
    ax.plot(radii, L_r[name], color="gray", linewidth=1.2, alpha=0.4, marker="s", markersize=5)

# R175H：加粗，带拟合线和平台标注
ax.plot(radii, L_r["R175H"], color=colors["R175H"], linewidth=2.5, marker="o", markersize=8, label="R175H")
r_fine = np.linspace(2, 30, 100)
ax.plot(r_fine, 22.7*np.exp(-0.107*r_fine) + c_vals[0],
         color=colors["R175H"], linestyle=":", linewidth=1.8, alpha=0.7)
ax.axhline(y=c_vals[0], color=colors["R175H"], linestyle="--", linewidth=1.2, alpha=0.4)
ax.annotate(f"c = {c_vals[0]:.1f}", xy=(28, c_vals[0]), fontsize=10, color=colors["R175H"])

ax.set_ylim(2, 20)
ax.set_xlim(0, 33)
ax.set_xlabel("Window Radius (residues)")
ax.set_ylabel("Relative Residual $L_r$")
ax.set_title("Convergence Dynamics Reveal Mutation-Specific Irreducible Plateaus", fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.3)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("Figures/fig3_convergence.pdf")
print("fig3_convergence.pdf saved.")
