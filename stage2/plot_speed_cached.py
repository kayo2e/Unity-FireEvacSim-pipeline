"""
plot_speed_cached.py — 포스터용 추론 속도 비교 차트
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager

_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM):
    font_manager.fontManager.addfont(_NANUM)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_NANUM).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

_ROOT   = os.path.dirname(os.path.abspath(__file__))
TRAIN_N = 40
PLOT_N  = [20, 40, 80, 120, 160, 200]

astar_time = [6.0, 12.0, 24.0, 36.0, 48.0, 62.0]
ppo_time   = [1.5, 1.5,  1.5,  1.5,  1.5,  1.63]

astar_surv = [82.0, 62.0, 63.0, 62.0, 60.0, 55.0]
ppo_surv   = [80.0, 72.0, 63.0, 58.0, 56.0, 51.0]

avg_ppo  = float(np.mean(ppo_surv))
avg_ast  = float(np.mean(astar_surv))
avg_diff = avg_ppo - avg_ast
n_ep = 30

# ── 스타일 ────────────────────────────────────────────────
BLUE      = "#1565C0"
RED       = "#C62828"
GRAY_LINE = "#BBBBBB"
GRAY_TEXT = "#555555"
BG_PANEL  = "white"

fig, ax = plt.subplots(figsize=(14.0, 8.5), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor(BG_PANEL)
for spine in ax.spines.values():
    spine.set_color("#CCCCCC")
    spine.set_linewidth(0.8)
ax.tick_params(colors="#444444", length=3, width=0.7)
ax.grid(True, color="white", linewidth=0.9, alpha=0.9, zorder=0)

# ── 선 ───────────────────────────────────────────────────
ax.plot(PLOT_N, astar_time, color=RED,  lw=2.2, marker="^",
        markersize=7, markerfacecolor=RED, markeredgecolor="white",
        markeredgewidth=0.8, label="A* 추론 시간", zorder=3)
ax.plot(PLOT_N, ppo_time,   color=BLUE, lw=2.2, marker="s",
        markersize=7, markerfacecolor=BLUE, markeredgecolor="white",
        markeredgewidth=0.8, label="PPO 추론 시간", zorder=3)

# ── 축 설정 ───────────────────────────────────────────────
max_t = max(astar_time) * 1.22
ax.set_ylim(0, max_t)
ax.set_xlim(15, 210)

ax.set_xlabel("인원 수", fontsize=27.5, color="#333333", labelpad=6)
ax.set_ylabel("추론 시간 (ms)", fontsize=27.5, color="#333333", labelpad=6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}ms"))
ax.tick_params(axis="both", labelsize=22.5)

x_labels = []
for n in PLOT_N:
    ratio = n / TRAIN_N
    frac  = f"×{int(ratio)}" if ratio == int(ratio) else f"×{ratio:.1f}"
    x_labels.append(f"{n}명\n({frac})")
ax.set_xticks(PLOT_N)
ax.set_xticklabels(x_labels, fontsize=21.25)

# ── 학습 밀도 수직선 ───────────────────────────────────────
ax.axvline(TRAIN_N, color="#888888", lw=1.2, ls="--", alpha=0.7, zorder=1)
ax.text(TRAIN_N + 2.5, max_t * 0.04, "학습 밀도\n(40명)",
        fontsize=18.75, color=GRAY_TEXT, va="bottom", linespacing=1.4)

# ── 배속 어노테이션 ────────────────────────────────────────
for n, at, pt in zip(PLOT_N[-2:], astar_time[-2:], ppo_time[-2:]):
    speedup = at / pt
    ax.annotate(
        f"{speedup:.0f}×",
        xy=(n, at), xytext=(n + 4, at - max_t * 0.08),
        fontsize=26.25, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2),
        zorder=7,
    )

# ── 범례 ─────────────────────────────────────────────────
leg = ax.legend(loc="lower right", fontsize=23.75,
                framealpha=0.95, facecolor="white",
                edgecolor="#CCCCCC", borderpad=0.7,
                handlelength=2.2, handletextpad=0.6)

# ── 저장 ─────────────────────────────────────────────────
out_dir  = os.path.join(_ROOT, "result", "density_extrapolation")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "density_s4_ep30_cached.png")

plt.tight_layout(pad=1.0)
fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=150)
plt.close(fig)
print(f"저장 완료: {out_path}")