"""
Plot training curves for the LaTeX report.
  - Blue solid line  : Curriculum agent (cpu_curriculum_run1)
  - Red dashed line  : Cold-start vs CEIA baseline (cpu_train_run1, longest trial)
Both smoothed with a rolling mean (window=20).
Horizontal dotted lines at ±2 mark the reward bounds.
Output: training_curve.pdf  (3.5 × 2.5 in, 300 dpi)
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

CURRICULUM_CSV = os.path.join(
    BASE,
    "ray_results/cpu_curriculum_run1"
    "/PPO_Soccer_ae5ed_00000_0_2026-04-21_20-03-24"
    "/progress.csv",
)

COLDSTART_CSV = os.path.join(
    BASE,
    "ray_results/cpu_train_run1"
    "/PPO_Soccer_44689_00000_0_2026-04-23_16-30-28"
    "/progress.csv",
)

OUT_PDF = os.path.join(BASE, "training_curve.pdf")

# ── load data ────────────────────────────────────────────────────────────────
COLS = ["timesteps_total", "episode_reward_mean"]

cur = pd.read_csv(CURRICULUM_CSV, usecols=COLS).dropna()
cold = pd.read_csv(COLDSTART_CSV,  usecols=COLS).dropna()

# convert steps → millions
cur["steps_M"]  = cur["timesteps_total"]  / 1e6
cold["steps_M"] = cold["timesteps_total"] / 1e6

# rolling mean (window=20, min_periods=1 so the first rows still show)
W = 20
cur["smooth"]  = cur["episode_reward_mean"].rolling(W, min_periods=1).mean()
cold["smooth"] = cold["episode_reward_mean"].rolling(W, min_periods=1).mean()

# ── plot ─────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(3.5, 2.5))

# raw signal as very faint fill so the smoothed line pops
ax.fill_between(cur["steps_M"],  cur["episode_reward_mean"],
                alpha=0.08, color="#1f77b4")
ax.fill_between(cold["steps_M"], cold["episode_reward_mean"],
                alpha=0.08, color="#d62728")

# smoothed lines
ax.plot(cur["steps_M"],  cur["smooth"],
        color="#1f77b4", lw=1.6, label="Curriculum (ours)")
ax.plot(cold["steps_M"], cold["smooth"],
        color="#d62728", lw=1.4, ls="--", label="Cold-start vs. CEIA")

# reference lines
ax.axhline(+2.0, color="gray", lw=0.8, ls=":", zorder=0)
ax.axhline(-2.0, color="gray", lw=0.8, ls=":", zorder=0)
ax.text(cold["steps_M"].max() * 0.98, 2.05, "+2 (max)",
        ha="right", va="bottom", fontsize=6, color="gray")
ax.text(cold["steps_M"].max() * 0.98, -2.15, "−2 (min)",
        ha="right", va="bottom", fontsize=6, color="gray")

# axes
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("Training Steps (millions)", fontsize=8)
ax.set_ylabel("Episode Reward Mean",       fontsize=8)
ax.tick_params(labelsize=7)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

# legend
ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

fig.tight_layout(pad=0.4)
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
print(f"Saved → {OUT_PDF}")
