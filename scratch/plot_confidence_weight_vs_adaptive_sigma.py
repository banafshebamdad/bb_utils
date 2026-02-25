#
# File: plot_confidence_weight_vs_adaptive_sigma.py
# Author: Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.6)
# Created: 2026-02-25 CET
#
"""
Comparison: Confidence as Heatmap Weight vs. Confidence Influencing Sigma.

Two approaches to incorporating InCrowd-VI confidence scores into GT heatmap
generation are shown side-by-side using a shared set of synthetic keypoints
with varying confidence levels.

Approach 1 — Confidence as amplitude weight
    H(x) = max_i [ c_i * G(x; u_i, σ_fixed) ]

    The Gaussian spread is identical for every keypoint.  Confidence
    scales the peak amplitude, so unreliable keypoints contribute less
    intensity but still occupy the same spatial footprint.

Approach 2 — Confidence-adaptive sigma
    σ_i = σ_min + (1 − c_i) · (σ_max − σ_min)
    H(x) = max_i [ G(x; u_i, σ_i) ]

    High-confidence keypoints get a narrow (sharp) Gaussian; low-confidence
    keypoints get a wide (diffuse) Gaussian, spreading their uncertainty over
    a larger neighbourhood.  The amplitude is not modulated.

Layout (4 rows × 2 columns)
    Row 1 : Individual Gaussian contributions (one curve per keypoint)
    Row 2 : Aggregated heatmap (max-of-Gaussians envelope)
    Row 3 : 2-D heatmap image
    Row 4 : Difference image  (Approach 2 − Approach 1)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Synthetic keypoints  (pixel coordinate, confidence)
# ---------------------------------------------------------------------------
KEYPOINTS = [
    # label       u     v     confidence
    ("KP-A",     40,   50,   0.92),   # reliable, isolated
    ("KP-B",     68,   47,   0.60),   # moderate, near KP-A
    ("KP-C",     72,   52,   0.25),   # uncertain, close to KP-B
    ("KP-D",     30,   75,   0.85),   # reliable, lower region
    ("KP-E",     80,   80,   0.15),   # very uncertain
]

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 120, 120
SIGMA_FIXED = 4.0          # constant sigma for Approach 1
SIGMA_MIN   = 2.5          # sigma for a perfectly confident keypoint (c=1)
SIGMA_MAX   = 8.0          # sigma for a perfectly uncertain keypoint (c=0)

COLORMAP_HEATMAP = "inferno"
COLORMAP_DIFF    = "RdBu_r"

KP_COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def adaptive_sigma(c: float) -> float:
    """Map confidence ∈ [0,1] to sigma.  Higher c → smaller sigma."""
    return SIGMA_MIN + (1.0 - c) * (SIGMA_MAX - SIGMA_MIN)


def gaussian_1d(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def gaussian_2d(Y: np.ndarray, X: np.ndarray, u: float, v: float,
                sigma: float) -> np.ndarray:
    return np.exp(-(((X - u) ** 2 + (Y - v) ** 2) / (2.0 * sigma ** 2)))


# ---------------------------------------------------------------------------
# 1-D slice along u-axis (v fixed at image centre)
# ---------------------------------------------------------------------------
x1d = np.linspace(0, IMG_W, 500)

# Approach 1: weighted, fixed sigma
gaussians_1d_weighted   = []
for _, u, v, c in KEYPOINTS:
    g = c * gaussian_1d(x1d, u, SIGMA_FIXED)
    gaussians_1d_weighted.append(g)
H1d_weighted = np.max(np.stack(gaussians_1d_weighted), axis=0)

# Approach 2: adaptive sigma, no amplitude weight
gaussians_1d_adaptive   = []
for _, u, v, c in KEYPOINTS:
    sig = adaptive_sigma(c)
    g = gaussian_1d(x1d, u, sig)
    gaussians_1d_adaptive.append(g)
H1d_adaptive = np.max(np.stack(gaussians_1d_adaptive), axis=0)

# ---------------------------------------------------------------------------
# 2-D heatmaps
# ---------------------------------------------------------------------------
ys = np.arange(IMG_H)
xs = np.arange(IMG_W)
X2, Y2 = np.meshgrid(xs, ys)

layers_weighted = []
layers_adaptive = []
for _, u, v, c in KEYPOINTS:
    layers_weighted.append(c * gaussian_2d(Y2, X2, u, v, SIGMA_FIXED))
    layers_adaptive.append(gaussian_2d(Y2, X2, u, v, adaptive_sigma(c)))

H2d_weighted = np.max(np.stack(layers_weighted), axis=0)
H2d_adaptive = np.max(np.stack(layers_adaptive), axis=0)
H2d_diff     = H2d_adaptive - H2d_weighted          # positive = adaptive stronger

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 18))
fig.suptitle(
    "Confidence as Heatmap Weight  vs.  Confidence-Adaptive Sigma",
    fontsize=14, fontweight="bold", y=0.995,
)

# 4 rows, 2 columns; extra height for the bottom row (difference)
gs = gridspec.GridSpec(
    4, 2,
    figure=fig,
    height_ratios=[1.1, 1.1, 1.2, 1.0],
    hspace=0.45,
    wspace=0.35,
)

col_titles = [
    "Approach 1 — Confidence as amplitude weight\n"
    r"$H(\mathbf{x}) = \max_i\;[c_i \cdot G(\mathbf{x};\,\mathbf{u}_i,\,\sigma_\mathrm{fixed})]$",
    "Approach 2 — Confidence-adaptive sigma\n"
    r"$\sigma_i = \sigma_\mathrm{min} + (1-c_i)(\sigma_\mathrm{max}-\sigma_\mathrm{min}),"
    r"\quad H(\mathbf{x}) = \max_i\;[G(\mathbf{x};\,\mathbf{u}_i,\,\sigma_i)]$",
]

# ── Row 0: individual 1-D Gaussians ─────────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])

for ax, gaussians, title in (
    (ax00, gaussians_1d_weighted, col_titles[0]),
    (ax01, gaussians_1d_adaptive, col_titles[1]),
):
    for idx, ((label, u, v, c), g, col) in enumerate(
        zip(KEYPOINTS, gaussians, KP_COLORS)
    ):
        sigma_str = (
            f"σ={SIGMA_FIXED:.1f}" if ax is ax00
            else f"σ={adaptive_sigma(c):.1f}"
        )
        ax.plot(x1d, g, color=col, linewidth=1.8,
                label=f"{label}  c={c:.2f}  {sigma_str}")
        ax.axvline(u, color=col, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel("Pixel position u", fontsize=8)
    ax.set_ylabel("Gaussian response", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right")

# ── Row 1: aggregated 1-D envelope ──────────────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1], sharey=ax10)

for ax, H1d, gaussians, title in (
    (ax10, H1d_weighted, gaussians_1d_weighted, "Aggregated heatmap (1-D slice)"),
    (ax11, H1d_adaptive, gaussians_1d_adaptive, "Aggregated heatmap (1-D slice)"),
):
    # shade individual contributions lightly
    for g, col in zip(gaussians, KP_COLORS):
        ax.fill_between(x1d, g, alpha=0.12, color=col)
        ax.plot(x1d, g, color=col, linewidth=0.9, alpha=0.5)
    # bold envelope
    ax.plot(x1d, H1d, color="#2c3e50", linewidth=2.2, label="max envelope")
    for idx, (label, u, v, c) in enumerate(KEYPOINTS):
        ax.axvline(u, color=KP_COLORS[idx], linestyle="--",
                   linewidth=0.9, alpha=0.7)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Pixel position u", fontsize=8)
    ax.set_ylabel("Heatmap value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

# ── Row 2: 2-D heatmap images ───────────────────────────────────────────────
ax20 = fig.add_subplot(gs[2, 0])
ax21 = fig.add_subplot(gs[2, 1])

vmin_h, vmax_h = 0.0, 1.0
for ax, H2d, title in (
    (ax20, H2d_weighted, "2-D heatmap — confidence weight"),
    (ax21, H2d_adaptive, "2-D heatmap — adaptive σ"),
):
    im = ax.imshow(H2d, origin="lower", cmap=COLORMAP_HEATMAP,
                   vmin=vmin_h, vmax=vmax_h, aspect="equal")
    # overlay keypoints
    for idx, (label, u, v, c) in enumerate(KEYPOINTS):
        sigma_disp = SIGMA_FIXED if ax is ax20 else adaptive_sigma(c)
        circle = plt.Circle((u, v), sigma_disp, color=KP_COLORS[idx],
                             fill=False, linewidth=1.5, linestyle="--")
        ax.add_patch(circle)
        ax.scatter(u, v, color=KP_COLORS[idx], s=55, zorder=5,
                   edgecolors="white", linewidths=0.7)
        ax.text(u + 2, v + 2, f"{label}\nc={c}", fontsize=6, color="white",
                va="bottom")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(0, IMG_H)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("u (px)", fontsize=8)
    ax.set_ylabel("v (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

# Dashed circle legend note
for ax in (ax20, ax21):
    ax.text(
        0.02, 0.02,
        "dashed circle = ±σ radius",
        transform=ax.transAxes,
        fontsize=6.5, color="white", alpha=0.85,
        va="bottom",
    )

# ── Row 3: difference image ──────────────────────────────────────────────────
ax30 = fig.add_subplot(gs[3, :])   # spans both columns

diff_abs_max = np.max(np.abs(H2d_diff))
im_diff = ax30.imshow(
    H2d_diff, origin="lower", cmap=COLORMAP_DIFF,
    norm=Normalize(vmin=-diff_abs_max, vmax=diff_abs_max),
    aspect="equal",
)
for idx, (label, u, v, c) in enumerate(KEYPOINTS):
    ax30.scatter(u, v, color=KP_COLORS[idx], s=60, zorder=5,
                 edgecolors="white", linewidths=0.8)
    ax30.text(u + 1.5, v + 1.5, label, fontsize=6.5, color="white")
ax30.set_xlim(0, IMG_W)
ax30.set_ylim(0, IMG_H)
ax30.set_title(
    "Difference: Adaptive σ − Confidence weight  "
    "(red = adaptive σ stronger, blue = confidence weight stronger)",
    fontsize=9,
)
ax30.set_xlabel("u (px)", fontsize=8)
ax30.set_ylabel("v (px)", fontsize=8)
ax30.tick_params(labelsize=7)
cb = plt.colorbar(im_diff, ax=ax30, fraction=0.02, pad=0.02)
cb.ax.tick_params(labelsize=7)
cb.set_label("Δ heatmap value", fontsize=7)

# ---------------------------------------------------------------------------
# Keypoint summary table (text box)
# ---------------------------------------------------------------------------
table_lines = [
    r"$\bf{Keypoint}$" + r"   $c$   " + r"$\sigma_1$(fixed)   " +
    r"$\sigma_2$(adaptive)",
]
for label, u, v, c in KEYPOINTS:
    sig2 = adaptive_sigma(c)
    table_lines.append(
        f"  {label:<6}  c={c:.2f}   σ₁={SIGMA_FIXED:.1f}   σ₂={sig2:.2f}"
    )
table_text = "\n".join(table_lines)

fig.text(
    0.5, -0.005,
    table_text,
    ha="center", va="top",
    fontsize=7.5,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.9),
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_path = "plot_confidence_weight_vs_adaptive_sigma.png"
plt.savefig(output_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {output_path}")
