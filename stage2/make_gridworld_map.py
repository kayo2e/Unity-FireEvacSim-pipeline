"""
make_gridworld_map.py — 포스터용 그리드월드 단일 맵 시각화
S1 시나리오, 에이전트 없이 건물 구조 + 화재만 표시
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM):
    font_manager.fontManager.addfont(_NANUM)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_NANUM).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_core import (
    FireEvacEnv, SCENARIO_CONFIGS,
    HALL, WALL, EXIT, ROOM,
    EXIT_A_POS, EXIT_B_POS, BASE_GRID,
)
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))

_VISUAL_WALL_CELLS = frozenset(
    (r, c) for r in range(10, 28) for c in range(8, 13)
    if BASE_GRID[r, c] == ROOM
) | frozenset(
    (r, c) for r in range(0, 10) for c in range(18, 25)
    if BASE_GRID[r, c] == HALL
)

EXIT_A_SET = set(EXIT_A_POS)
EXIT_B_SET = set(EXIT_B_POS)

COLORS = {
    "hall":    "#F0F0F0",
    "wall":    "#2C2C2C",
    "exit":    "#00A550",
    "room":    "#E8DFC8",
    "fire":    "#FF4500",
    "smoke":   "#C8C8C8",
    "blocked": "#7A0000",
}

def _hex2rgba(h, a=1.0):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)] + [a]


def _load_exit_icon(size=36):
    path = os.path.join(_ROOT, "evac_sign.png")
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    return np.array(img)


def render_map(env, ax, step: int):
    ROWS, COLS = env.ROWS, env.COLS
    img = np.zeros((ROWS, COLS, 4), dtype=np.float32)

    for r in range(ROWS):
        for c in range(COLS):
            cell = env.grid[r, c]
            if cell == WALL or (r, c) in _VISUAL_WALL_CELLS:
                img[r, c] = _hex2rgba(COLORS["wall"])
            elif (r, c) in EXIT_A_SET:
                col = COLORS["blocked"] if (r, c) in env.blocked_exits else COLORS["exit"]
                img[r, c] = _hex2rgba(col)
            elif (r, c) in EXIT_B_SET:
                col = COLORS["blocked"] if (r, c) in env.blocked_exits else COLORS["exit"]
                img[r, c] = _hex2rgba(col)
            elif cell == ROOM:
                img[r, c] = _hex2rgba(COLORS["room"])
            else:
                img[r, c] = _hex2rgba(COLORS["hall"])

    # 화재
    for r, c in zip(*np.where(env.fire_map > 0)):
        img[r, c] = _hex2rgba(COLORS["fire"], 0.95)

    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    # ── 화재 글로우 ──
    fire_ys, fire_xs = np.where(env.fire_map > 0)
    if len(fire_xs):
        is_wall = np.array([
            env.grid[r, c] == WALL or (r, c) in _VISUAL_WALL_CELLS
            for r, c in zip(fire_ys, fire_xs)
        ])
        frs = fire_ys[~is_wall].astype(float)
        fcs = fire_xs[~is_wall].astype(float)
        if len(fcs):
            ax.scatter(fcs, frs, s=260, c="#CC1000", alpha=0.10, linewidths=0, zorder=3)
            ax.scatter(fcs, frs, s=140, c="#FF5000", alpha=0.22, linewidths=0, zorder=3)
            ax.scatter(fcs, frs, s=60,  c="#FF9500", alpha=0.55, linewidths=0, zorder=3)
            ax.scatter(fcs, frs, s=22,  c="#FFE040", alpha=0.85, linewidths=0, zorder=3)
            ax.scatter(fcs, frs - 0.45, s=90, c="#FF6A00", alpha=0.50,
                       marker="^", linewidths=0, zorder=3.5)

    # ── 출구 아이콘 ──
    icon = _load_exit_icon(size=40)
    for exit_set in (EXIT_A_SET, EXIT_B_SET):
        rows = sorted({r for r, c in exit_set if (r, c) not in env.blocked_exits})
        if not rows:
            continue
        mid_r = rows[len(rows)//2]
        mid_c = next(c for r, c in exit_set if r == mid_r)
        ib = OffsetImage(icon, zoom=0.55)
        ab = AnnotationBbox(ib, (mid_c, mid_r), frameon=False, zorder=9)
        ax.add_artist(ab)

    # ── 격자선 (얇게) ──
    for r in range(ROWS + 1):
        ax.axhline(r - 0.5, color="#AAAAAA", lw=0.25, alpha=0.35, zorder=1)
    for c in range(COLS + 1):
        ax.axvline(c - 0.5, color="#AAAAAA", lw=0.25, alpha=0.35, zorder=1)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#888888")
        sp.set_linewidth(1.2)

    # step 표시
    ax.text(0.99, 0.01, f"t = {step}",
            transform=ax.transAxes, fontsize=11,
            ha="right", va="bottom", color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.80, edgecolor="#CCCCCC", linewidth=0.8))


# ── 메인 ─────────────────────────────────────────────────
SCENARIO  = 1
TARGET    = 70
SEED      = 42

np.random.seed(SEED)
env = FireEvacEnv(scenario=SCENARIO, n_agents=40)
env.reset(seed=SEED)
for _ in range(TARGET):
    env.step([0] * env.n_agents)
actual = min(TARGET, env.step_count)

fig, ax = plt.subplots(figsize=(4.5, 7.2), dpi=180)
fig.patch.set_facecolor("white")
render_map(env, ax, actual)
env.close()

# ── 범례 ──
legend_items = [
    mpatches.Patch(facecolor=COLORS["hall"],  edgecolor="#AAAAAA", lw=0.6, label="복도"),
    mpatches.Patch(facecolor=COLORS["room"],  edgecolor="#AAAAAA", lw=0.6, label="방"),
    mpatches.Patch(facecolor=COLORS["wall"],  label="벽"),
    mpatches.Patch(facecolor=COLORS["exit"],  label="출구"),
    mpatches.Patch(facecolor=COLORS["fire"],  label="화재"),
]
fig.legend(handles=legend_items, loc="lower center",
           ncol=5, fontsize=8.5,
           framealpha=0.95, edgecolor="#CCCCCC",
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 1])

out_dir  = os.path.join(_ROOT, "result", "visualize")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"gridworld_s{SCENARIO}_t{actual}.png")
fig.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=180)
plt.close(fig)
print(f"저장 완료: {out_path}")