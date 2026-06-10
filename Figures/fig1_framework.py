import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 9, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
})

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")

# ===== 1. 全局序列 =====
seq_box = FancyBboxPatch((0.3, 3.0), 2.0, 1.0, boxstyle="round,pad=0.15",
                         facecolor="#E3F2FD", edgecolor="#1F77B4", linewidth=1.5)
ax.add_patch(seq_box)
ax.text(1.3, 3.8, "Full Sequence", ha="center", va="center", fontweight="bold", color="#1F77B4")
ax.text(1.3, 3.3, "(Global Context)", ha="center", va="center", fontsize=8, color="#1F77B4")

# ===== 2. ProtT5 / ESM-2 =====
plm_box = FancyBboxPatch((3.0, 3.0), 2.2, 1.0, boxstyle="round,pad=0.15",
                         facecolor="#FFF3E0", edgecolor="#FF7F0E", linewidth=1.5)
ax.add_patch(plm_box)
ax.text(4.1, 3.8, "Protein Language", ha="center", va="center", fontweight="bold", color="#FF7F0E")
ax.text(4.1, 3.5, "Model", ha="center", va="center", fontweight="bold", color="#FF7F0E")
ax.text(4.1, 3.2, "(ProtT5 / ESM-2)", ha="center", va="center", fontsize=8, color="#FF7F0E")

# ===== 3. 全局嵌入 =====
emb_box = FancyBboxPatch((6.0, 3.0), 2.4, 1.0, boxstyle="round,pad=0.15",
                         facecolor="#E8F5E9", edgecolor="#2CA02C", linewidth=1.5)
ax.add_patch(emb_box)
ax.text(7.2, 3.8, "Residue Embeddings", ha="center", va="center", fontweight="bold", color="#2CA02C")
ax.text(7.2, 3.3, r"$\mathbf{H} \in \mathbb{R}^{L \times d}$", ha="center", va="center", fontsize=10, color="#2CA02C")

# ===== 箭头：全局序列 → PLM → 嵌入 =====
ax.annotate("", xy=(3.0, 3.5), xytext=(2.3, 3.5),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.8))
ax.annotate("", xy=(6.0, 3.5), xytext=(5.2, 3.5),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.8))

# ===== 4. 局部窗口 =====
win_box = FancyBboxPatch((3.0, 1.5), 2.2, 0.9, boxstyle="round,pad=0.15",
                         facecolor="#F3E5F5", edgecolor="#9467BD", linewidth=1.5)
ax.add_patch(win_box)
ax.text(4.1, 2.2, "Local Window", ha="center", va="center", fontweight="bold", color="#9467BD")
ax.text(4.1, 1.8, r"($\pm r$ residues)", ha="center", va="center", fontsize=8, color="#9467BD")

# ===== 箭头：嵌入 → 局部窗口 =====
ax.annotate("", xy=(3.6, 2.4), xytext=(5.5, 2.95),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, connectionstyle="arc3,rad=-0.3"))

# ===== 5. 局部编码器 =====
enc_box = FancyBboxPatch((6.0, 1.5), 2.4, 0.9, boxstyle="round,pad=0.15",
                         facecolor="#FFEBEE", edgecolor="#D62728", linewidth=1.5)
ax.add_patch(enc_box)
ax.text(7.2, 2.2, "Local Encoder", ha="center", va="center", fontweight="bold", color="#D62728")
ax.text(7.2, 1.8, "(2-layer Transformer)", ha="center", va="center", fontsize=8, color="#D62728")

# ===== 箭头：局部窗口 → 编码器 =====
ax.annotate("", xy=(6.0, 1.95), xytext=(5.2, 1.95),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.8))

# ===== 6. 预测输出 =====
out_box = FancyBboxPatch((9.0, 1.5), 2.0, 0.9, boxstyle="round,pad=0.15",
                         facecolor="#E8F5E9", edgecolor="#2CA02C", linewidth=1.5)
ax.add_patch(out_box)
ax.text(10.0, 2.2, "Predicted Shift", ha="center", va="center", fontweight="bold", color="#2CA02C")
ax.text(10.0, 1.8, r"$\Delta \hat{h}_{\rm local}$", ha="center", va="center", fontsize=10, color="#2CA02C")

# ===== 箭头：编码器 → 预测 =====
ax.annotate("", xy=(9.0, 1.95), xytext=(8.4, 1.95),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.8))

# ===== 7. 真实输出（对比） =====
true_box = FancyBboxPatch((9.0, 0.3), 2.0, 0.9, boxstyle="round,pad=0.15",
                          facecolor="#E8F5E9", edgecolor="#2CA02C", linewidth=1.5, alpha=0.6)
ax.add_patch(true_box)
ax.text(10.0, 1.0, "Global Shift", ha="center", va="center", fontweight="bold", color="#2CA02C")
ax.text(10.0, 0.6, r"$\Delta h_{\rm global}$", ha="center", va="center", fontsize=10, color="#2CA02C")

# ===== 箭头：预测 → 真实（虚线） =====
ax.annotate("", xy=(9.0, 0.75), xytext=(9.0, 1.5),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dashed"))
ax.text(8.5, 1.1, "compare", ha="center", va="center", fontsize=8, color="gray")

# ===== 8. 误差分解 =====
met_box = FancyBboxPatch((11.5, 1.5), 2.3, 1.2, boxstyle="round,pad=0.15",
                         facecolor="#ECEFF1", edgecolor="#607D8B", linewidth=1.5)
ax.add_patch(met_box)
ax.text(12.65, 2.5, "Geometric", ha="center", va="center", fontweight="bold", color="#607D8B")
ax.text(12.65, 2.2, "Decomposition", ha="center", va="center", fontweight="bold", color="#607D8B")
ax.text(12.65, 1.9, r"$L_\theta$, $L_m$, $L_r$", ha="center", va="center", fontsize=9, color="#607D8B")

# ===== 箭头：预测/真实 → 度量 =====
ax.annotate("", xy=(11.5, 1.95), xytext=(11.0, 1.95),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.8))
ax.annotate("", xy=(11.5, 1.5), xytext=(11.0, 0.75),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, connectionstyle="arc3,rad=0.3"))

# ===== 9. 收敛分析 =====
conv_box = FancyBboxPatch((11.5, 0.1), 2.3, 1.2, boxstyle="round,pad=0.15",
                          facecolor="#ECEFF1", edgecolor="#607D8B", linewidth=1.5)
ax.add_patch(conv_box)
ax.text(12.65, 1.0, "Convergence", ha="center", va="center", fontweight="bold", color="#607D8B")
ax.text(12.65, 0.7, "Analysis", ha="center", va="center", fontweight="bold", color="#607D8B")
ax.text(12.65, 0.4, r"$L(r) \rightarrow c, \alpha$", ha="center", va="center", fontsize=9, color="#607D8B")

# ===== 箭头：度量 → 收敛 =====
ax.annotate("", xy=(12.65, 1.3), xytext=(12.65, 1.5),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

# ===== 标题 =====
ax.text(7.0, 4.7, "Local Encoder Framework for Probing Representation Locality",
        ha="center", va="center", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("Figures/fig1_framework.pdf")
print("fig1_framework.pdf saved.")
