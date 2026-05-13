"""
Fig 2 — Autoregressive MDP Formulation + Neural Network
실행: python draw_fig2_method.py
출력: fig2_method.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── 색상 ──────────────────────────────────────────
C_PROBLEM   = "#FFF8E1"
C_TRAD      = "#FFCCBC"
C_AR        = "#C8E6C9"
C_OBS       = "#E3F2FD"
C_CELL      = "#E8EAF6"
C_GLOBAL    = "#FFF9C4"
C_POS       = "#FCE4EC"
C_FC        = "#E8F5E9"
C_LSTM      = "#EDE7F6"
C_ACTOR     = "#FFCDD2"
C_CRITIC    = "#E0F7FA"
C_BORDER    = "#90A4AE"


def box(x, y, w, h, fc, ec=C_BORDER, lw=1.5, r=0.12):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc))


def txt(x, y, s, fs=9, fw="normal", c="black", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=c,
            ha=ha, va=va, fontstyle=style, linespacing=1.4)


def arr(x1, y1, x2, y2, c="#546E7A", lw=1.6, label="", ldy=0.17):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw, mutation_scale=14))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+ldy, label, fontsize=7.5, color="#37474F",
                ha="center", style="italic")


# ══════════════════════════════════════════════════════
# SECTION A — Problem Formulation  (x: 0.3 ~ 6.0)
# ══════════════════════════════════════════════════════
box(0.2, 0.5, 5.8, 9.0, C_PROBLEM, ec="#F57F17", lw=2)
txt(3.1, 9.27, "Problem Formulation", fs=11, fw="bold", c="#E65100")

# ── A1: Traditional (위쪽) ──
box(0.5, 6.6, 5.2, 2.5, C_TRAD, ec="#BF360C", lw=1.5)
txt(3.1, 8.9, "Traditional Multi-Agent", fs=9, fw="bold", c="#BF360C")

# 에이전트들 동시 결정
for i in range(5):
    bx = 0.7 + i * 0.98
    box(bx, 7.45, 0.82, 0.55, "white", ec="#BF360C", lw=1)
    txt(bx+0.41, 7.72, f"A{i+1}", fs=8, c="#BF360C", fw="bold")

# joint action 화살표
ax.annotate("", xy=(3.1, 6.9), xytext=(3.1, 7.45),
            arrowprops=dict(arrowstyle="->", color="#BF360C", lw=1.5))
box(1.8, 6.62, 2.6, 0.55, "#FFAB91", ec="#BF360C", lw=1.2)
txt(3.1, 6.89, "Joint Action  |A|ᴺ = 2⁴⁰ = 10¹²", fs=8.5, fw="bold", c="#B71C1C")
txt(3.1, 7.15, "All agents decide simultaneously", fs=8, c="#5D4037", style="italic")

# 빨간 X
txt(3.1, 6.35, "✗  Intractable action space", fs=9, c="#C62828", fw="bold")

# ── A2: Autoregressive (아래쪽) ──
box(0.5, 1.0, 5.2, 5.2, C_AR, ec="#1B5E20", lw=1.8)
txt(3.1, 6.0, "Autoregressive Decomposition  (Ours)", fs=9, fw="bold", c="#1B5E20")

# 1 tick = K sequential steps 설명
box(0.7, 5.45, 4.8, 0.45, "#F1F8E9", ec="#388E3C", lw=1)
txt(3.1, 5.67, "1 simulation tick  →  K_MAX sequential gym steps", fs=8.5, c="#2E7D32")

# 순서 화살표 체인
ys = [4.65, 3.85, 3.05]
labels = ["Agent i₁  →  obs₁  →  action₁",
          "Agent i₂  →  obs₂  →  action₂",
          "      ⋮         ⋮          ⋮"]
colors_ag = ["#A5D6A7", "#C8E6C9", "#E8F5E9"]
for i, (y, label, fc) in enumerate(zip(ys, labels, colors_ag)):
    box(0.7, y, 4.8, 0.6, fc, ec="#43A047", lw=1)
    txt(3.1, y+0.3, label, fs=8.5, c="#1B5E20")

# LSTM 상태 전달 표시
for y in [4.65, 3.85]:
    ax.annotate("", xy=(0.9, y), xytext=(0.9, y+0.6),
                arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.4))
txt(0.58, 3.98, "LSTM\nstate", fs=7, c="#7B1FA2", ha="center")

# single action
box(1.5, 1.35, 3.2, 0.65, "#B9F6CA", ec="#00C853", lw=1.5)
txt(3.1, 1.67, "Single Action  Discrete(2)  =  2", fs=9, fw="bold", c="#00695C")
txt(3.1, 1.12, "✓  Tractable  |  Agents guided individually", fs=8.5, c="#1B5E20", fw="bold")

arr(3.1, 3.05, 3.1, 2.0, c="#388E3C", lw=1.5)


# ══════════════════════════════════════════════════════
# SECTION B — Observation (25-dim)  (x: 6.4 ~ 11.2)
# ══════════════════════════════════════════════════════
box(6.3, 0.5, 5.0, 9.0, C_OBS, ec="#1565C0", lw=2)
txt(8.8, 9.27, "Observation  (25-dim per step)", fs=11, fw="bold", c="#0D47A1")

# ── Cell features (8) ──
box(6.55, 5.6, 4.5, 3.55, C_CELL, ec="#3949AB", lw=1.5)
txt(8.8, 8.97, "Cell Features  (8-dim)", fs=9.5, fw="bold", c="#283593")

cell_feats = [
    ("fire_near",    "Nearby fire presence"),
    ("smoke_near",   "Nearby smoke presence"),
    ("density",      "Local crowd density"),
    ("active_flag",  "Slot occupied"),
    ("dist_A",       "BFS dist to Exit A"),
    ("dist_B",       "BFS dist to Exit B"),
    ("row",          "Cell row (normalized)"),
    ("col",          "Cell col (normalized)"),
]
for i, (name, desc) in enumerate(cell_feats):
    y = 8.55 - i * 0.37
    box(6.65, y-0.14, 4.3, 0.32, "white", ec="#9FA8DA", lw=0.8)
    txt(7.3, y+0.02, name, fs=7.5, fw="bold", c="#1A237E", ha="left")
    txt(10.3, y+0.02, desc, fs=7, c="#37474F", ha="right")

# ── Global features (15) ──
box(6.55, 2.2, 4.5, 3.1, C_GLOBAL, ec="#F57F17", lw=1.5)
txt(8.8, 5.12, "Global Features  (15-dim)  F1~F15", fs=9.5, fw="bold", c="#E65100")

global_feats = [
    ("F1/F2",    "Exit A/B fire threat"),
    ("F3",       "Ratio closer to Exit A"),
    ("F4/F5",    "Escaped / dead ratio"),
    ("F6",       "Time progress"),
    ("F7/F8",    "Exit A/B congestion"),
    ("F9",       "Mean panic index"),
    ("F10/F11",  "Mean BFS dist to exits"),
    ("F12/F13",  "Fire centroid position"),
    ("F14/F15",  "Fire approach speed"),
]
for i, (name, desc) in enumerate(global_feats):
    y = 4.8 - i * 0.28
    box(6.65, y-0.11, 4.3, 0.25, "white", ec="#FFE082", lw=0.7)
    txt(7.3, y+0.02, name, fs=7.2, fw="bold", c="#BF360C", ha="left")
    txt(10.3, y+0.02, desc, fs=7, c="#37474F", ha="right")

# ── Position features (2) ──
box(6.55, 0.85, 4.5, 1.1, C_POS, ec="#AD1457", lw=1.5)
txt(8.8, 1.77, "Position Features  (2-dim)", fs=9.5, fw="bold", c="#880E4F")
box(6.7, 0.9, 1.95, 0.4, "white", ec="#F48FB1", lw=0.8)
txt(7.68, 1.1, "slot_idx / K_MAX", fs=7.5, fw="bold", c="#880E4F")
box(8.75, 0.9, 1.95, 0.4, "white", ec="#F48FB1", lw=0.8)
txt(9.73, 1.1, "n_active / K_MAX", fs=7.5, fw="bold", c="#880E4F")

# A → B 화살표
arr(6.0, 5.0, 6.3, 5.0, c="#E65100", lw=2, label="obs₍ᵢ₎")


# ══════════════════════════════════════════════════════
# SECTION C — Neural Network  (x: 11.6 ~ 17.8)
# ══════════════════════════════════════════════════════
box(11.5, 0.5, 6.2, 9.0, "#F9FBE7", ec="#33691E", lw=2)
txt(14.6, 9.27, "RecurrentPPO  Policy Network", fs=11, fw="bold", c="#1B5E20")

# 입력
box(12.3, 7.8, 4.6, 0.7, C_OBS, ec="#1565C0", lw=1.5)
txt(14.6, 8.15, "Input: obs (25-dim)", fs=9.5, fw="bold", c="#0D47A1")

# FC layers
box(12.3, 6.6, 4.6, 0.7, C_FC, ec="#388E3C", lw=1.5)
txt(14.6, 6.95, "FC(256)  + ReLU", fs=9.5)

box(12.3, 5.4, 4.6, 0.7, C_FC, ec="#388E3C", lw=1.5)
txt(14.6, 5.75, "FC(256)  + ReLU", fs=9.5)

# LSTM
box(12.3, 4.0, 4.6, 0.9, C_LSTM, ec="#6A1B9A", lw=2)
txt(14.6, 4.45, "LSTM  (hidden_size = 256)", fs=10, fw="bold", c="#4A148C")

# recurrent 화살표
ax.annotate("", xy=(12.1, 4.45), xytext=(12.1, 5.4),
            arrowprops=dict(arrowstyle="<->", color="#7B1FA2", lw=1.8,
                           connectionstyle="arc3,rad=-0.5"))
txt(11.75, 4.9, "(h, c)", fs=8, c="#7B1FA2")

# 분기
arr(14.6, 4.0, 14.6, 3.3, c="#546E7A", lw=1.5)
arr(14.6, 3.3, 13.0, 3.3, c="#546E7A", lw=1.3)
arr(14.6, 3.3, 16.2, 3.3, c="#546E7A", lw=1.3)

# Actor
box(11.85, 1.8, 2.3, 1.2, C_ACTOR, ec="#C62828", lw=1.8)
txt(13.0, 2.7, "Actor", fs=10, fw="bold", c="#B71C1C")
txt(13.0, 2.3, "Discrete(2)", fs=8.5, c="#B71C1C")
txt(13.0, 1.97, "softmax", fs=8, c="#B71C1C", style="italic")
arr(13.0, 3.3, 13.0, 3.0, c="#C62828", lw=1.5)

# Critic
box(15.1, 1.8, 2.3, 1.2, C_CRITIC, ec="#00838F", lw=1.8)
txt(16.25, 2.4, "Critic", fs=10, fw="bold", c="#006064")
txt(16.25, 2.0, "V(s)", fs=9, c="#006064")
arr(16.25, 3.3, 16.25, 3.0, c="#00838F", lw=1.5)

# 출력 박스
box(11.85, 0.65, 2.3, 0.85, "#FFCDD2", ec="#E53935", lw=1.5)
txt(13.0, 1.07, "P(Exit A)\nP(Exit B)", fs=8.5, c="#B71C1C", fw="bold")
arr(13.0, 1.8, 13.0, 1.5, c="#C62828", lw=1.5)

box(15.1, 0.65, 2.3, 0.85, "#E0F7FA", ec="#00838F", lw=1.5)
txt(16.25, 1.07, "Value\nEstimate", fs=8.5, c="#006064", fw="bold")
arr(16.25, 1.8, 16.25, 1.5, c="#00838F", lw=1.5)

# 내부 화살표
arr(14.6, 7.8, 14.6, 7.3, lw=1.3)
arr(14.6, 6.6, 14.6, 6.1, lw=1.3)
arr(14.6, 5.4, 14.6, 4.9, lw=1.3)

# B → C 화살표
arr(11.3, 5.0, 11.5, 8.15, c="#1565C0", lw=2, label="25-dim")


# ══════════════════════════════════════════════════════
# 제목
# ══════════════════════════════════════════════════════
ax.text(9.0, 9.7,
        "Fig 2.  Autoregressive MDP Formulation and Policy Network",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#212121",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECEFF1",
                  edgecolor=C_BORDER, lw=1.5))

plt.tight_layout(pad=0.2)
out = "fig2_method.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
