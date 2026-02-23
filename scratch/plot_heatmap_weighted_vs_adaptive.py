#
# File: plot_heatmap_weighted_vs_adaptive.py
# Author: Banafshe Bamdad + ChatGPT 5.2
# Email: banafshebamdad@gmail.com
# Created: 2026-02-22 16:12 CET
#
"""
Compare weighted-Gaussian and adaptive-sigma heatmap aggregation in GT heatmap generation.

This script generates a 1D visualization illustrating how confidence
weighting and adaptive spatial uncertainty affect the resulting
supervision signal.

"""
# Generate a comparison figure between:
# 1) Weighted Gaussian + max
# 2) Adaptive sigma (no amplitude weighting)

import numpy as np
import matplotlib.pyplot as plt

# Keypoint positions (1D for clarity)
x = np.linspace(90, 110, 1000)

# Keypoint A (reliable)
u_A = 100
c_A = 0.9

# Keypoint B (uncertain)
u_B = 102
c_B = 0.3

# Fixed sigma for weighted case
sigma_fixed = 2.0

# Adaptive sigma parameters
sigma_min = 1.5
sigma_max = 3.0

def adaptive_sigma(c):
    return sigma_min + (1 - c) * (sigma_max - sigma_min)

sigma_A = adaptive_sigma(c_A)
sigma_B = adaptive_sigma(c_B)

# ---- Gaussian function ----
def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# ---- Case 1: Weighted + max ----
G_A_weighted = c_A * gaussian(x, u_A, sigma_fixed)
G_B_weighted = c_B * gaussian(x, u_B, sigma_fixed)
H_weighted = np.maximum(G_A_weighted, G_B_weighted)

# ---- Case 2: Adaptive sigma (no amplitude weighting) ----
G_A_adapt = gaussian(x, u_A, sigma_A)
G_B_adapt = gaussian(x, u_B, sigma_B)
H_adapt = np.maximum(G_A_adapt, G_B_adapt)

# ---- Plot ----
plt.figure(figsize=(8, 4.5))
plt.plot(x, H_weighted, label="Weighted Gaussian + max")
plt.plot(x, H_adapt, label="Adaptive sigma + max")
plt.axvline(u_A, linestyle="--", label="Keypoint A")
plt.axvline(u_B, linestyle=":", label="Keypoint B")

plt.title("Comparison: Weighted vs Adaptive Sigma Heatmaps")
plt.xlabel("Pixel position (1D slice)")
plt.ylabel("Heatmap response")
plt.legend()
plt.tight_layout()

output_path = "plot_heatmap_weighted_vs_adaptive.png"
plt.savefig(output_path, dpi=200)
plt.close()

output_path
