import numpy as np
from scipy.optimize import curve_fit

radii = np.array([2, 5, 10, 15, 20, 30])
L_r_data = {
    "R175H":  [17.99, 14.64, 10.31, 7.47, 5.25, 5.25],
    "G245S":  [15.10, 12.40, 8.87,  5.93, 4.34, 4.34],
    "R249S":  [11.97, 9.77,  6.90,  4.82, 3.47, 3.47],
    "R282W":  [12.49, 10.38, 7.55,  4.83, 3.47, 3.47],
    "Y220C":  [14.81, 12.37, 8.33,  5.70, 3.94, 3.94],
}

def exp_decay(r, A, alpha, c):
    return A * np.exp(-alpha * r) + c

print("Mutation    c_fitted    std_error    cv(%)")
print("-" * 50)
for name, L_r_vals in L_r_data.items():
    try:
        popt, pcov = curve_fit(exp_decay, radii, L_r_vals, p0=[10, 0.1, 3], maxfev=10000)
        c_fit = popt[2]
        c_std = np.sqrt(pcov[2, 2])
        cv = 100 * c_std / abs(c_fit)  # 变异系数
        print(f"{name:8s}   {c_fit:.3f}      ± {c_std:.3f}        {cv:.1f}%")
    except Exception as e:
        print(f"{name:8s}   拟合失败: {e}")
