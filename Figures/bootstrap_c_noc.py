import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
})

c_vals = np.array([4.015, 3.174, 2.623, 2.326, 2.739])
noc_vals = np.array([42, 32, 30, 36, 32])
n_boot = 10000
rng = np.random.default_rng(42)

boot_rhos = []
n_invalid = 0
for _ in range(n_boot):
    idx = rng.choice(len(c_vals), size=len(c_vals), replace=True)
    boot_c = c_vals[idx]
    boot_noc = noc_vals[idx]
    # 检查是否为常数数组
    if np.std(boot_c) == 0 or np.std(boot_noc) == 0:
        n_invalid += 1
        continue
    rho, _ = spearmanr(boot_c, boot_noc)
    boot_rhos.append(rho)

boot_rhos = np.array(boot_rhos)
mean_rho = np.mean(boot_rhos)
ci_lower = np.percentile(boot_rhos, 2.5)
ci_upper = np.percentile(boot_rhos, 97.5)

print(f"Valid bootstrap iterations: {len(boot_rhos)}/{n_boot} ({n_invalid} constant samples excluded)")
print(f"Bootstrap mean ρ: {mean_rho:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(boot_rhos, bins=50, color="gray", alpha=0.7, edgecolor="white")
ax.axvline(mean_rho, color="red", linestyle="--", linewidth=2, label=f"Mean ρ = {mean_rho:.2f}")
ax.axvline(ci_lower, color="blue", linestyle=":", linewidth=2, label=f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
ax.axvline(ci_upper, color="blue", linestyle=":", linewidth=2)
ax.set_xlabel("Spearman ρ")
ax.set_ylabel("Frequency")
ax.set_title("Bootstrap Distribution of c vs NOC Correlation")
ax.legend()
plt.tight_layout()
plt.savefig("Figures/figS4_bootstrap.pdf")
print("figS4_bootstrap.pdf saved.")
