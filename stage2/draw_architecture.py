"""
AutoregressivePPO 아키텍처 figure 생성
실행: python draw_architecture.py
출력: autoregressive_ppo_architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 색상 팔레트 ───────────────────────────────────────
C_ENV    = "#E3F2FD"   # 환경 (파랑 계열)
C_OBS    = "#FFF8E1"   # 관측 (노랑)
C_LSTM   = "#EDE7F6"   # LSTM (보라)
C_FC     = "#E8F5E9"   # FC layer (초록)
C_ACTOR  = "#FCE4EC"   # Actor (분홍)
C_CRITIC = "#E0F7FA"   # Critic (청록)
C_EXIT   = "#FFF3E0"   # 출구 신호
C_BORDER = "#90A4AE"

fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlim(0, 18)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, fc, ec=C_BORDER, lw=1.5, radius=0.15):
    p = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(p)
    return p


def label(x, y, txt, fs=9, fw="normal", color="black", ha="center", va="center"):
    ax.text(x, y, txt, fontsize=fs, fontweight=fw, color=color,
            ha=ha, va=va, linespacing=1.4)


def arrow(x1, y1, x2, y2, color="#546E7A", lw=1.5, txt="", txt_dy=0.18):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=14))
    if txt:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + txt_dy, txt, fontsize=7.5, color="#37474F",
                ha="center", va="bottom", style="italic")


# ════════════════════════════════════════════════
# (A) 화재 환경 그리드  x: 0.4~3.8
# ════════════════════════════════════════════════
box(0.3, 1.2, 3.6, 7.0, C_ENV, ec="#1565C0", lw=2)
label(2.1, 8.0, "Fire Evacuation\nEnvironment", fs=10, fw="bold", color="#1565C0")

# 미니 그리드 8×8
GX, GY, CS = 0.55, 2.0, 0.38
n = 8
fire_cells   = {(0,0),(0,1),(1,0),(1,1),(0,2)}
exit_a_cells = {(0,7)}
exit_b_cells = {(7,4)}
agent_cells  = {(3,3),(4,5),(5,2),(6,6),(2,5)}

for r in range(n):
    for c in range(n):
        if (r, c) in fire_cells:
            fc = "#EF5350"
        elif (r, c) in exit_a_cells:
            fc = "#66BB6A"
        elif (r, c) in exit_b_cells:
            fc = "#42A5F5"
        elif (r, c) in agent_cells:
            fc = "#FFA726"
        else:
            fc = "white"
        rect = mpatches.Rectangle(
            (GX + c * CS, GY + (n - 1 - r) * CS), CS, CS,
            lw=0.4, edgecolor="#BDBDBD", facecolor=fc
        )
        ax.add_patch(rect)

# 범례
items = [("#EF5350","Fire"), ("#FFA726","Agent"),
         ("#66BB6A","Exit A"), ("#42A5F5","Exit B")]
for i, (c, t) in enumerate(items):
    bx = 0.55 + i * 0.88
    ax.add_patch(mpatches.Rectangle((bx, 1.45), 0.22, 0.22,
                                     facecolor=c, lw=0, zorder=3))
    ax.text(bx + 0.28, 1.56, t, fontsize=7, va="center")

# ════════════════════════════════════════════════
# (B) Autoregressive 분해  x: 4.2~8.6
# ════════════════════════════════════════════════
box(4.1, 1.2, 4.6, 7.0, "#FFFDE7", ec="#F57F17", lw=2, radius=0.2)
label(6.4, 8.0, "Autoregressive Decomposition\n(1 Simulation Tick)", fs=10, fw="bold", color="#E65100")

label(6.4, 7.35, "1 tick  =  K_MAX gym steps (sequential)", fs=8.5, color="#BF360C")

# 에이전트 박스들 (slot 1~4 + 점점점)
slots = [
    ("Slot 1", "Agent  i₁", "#FFCCBC"),
    ("Slot 2", "Agent  i₂", "#FFE0B2"),
    ("Slot 3", "Agent  i₃", "#FFF9C4"),
    ("Slot 4", "Agent  i₄", "#DCEDC8"),
]
SX, SW, SH = 4.35, 3.8, 0.72
gaps = [6.4, 5.45, 4.5, 3.55]

for (slot, agent, fc), sy in zip(slots, gaps):
    box(SX, sy, SW, SH, fc, ec="#FB8C00", lw=1.2)
    label(SX + SW * 0.27, sy + SH / 2, slot, fs=8, fw="bold", color="#E65100")
    label(SX + SW * 0.72, sy + SH / 2,
          f"{agent}\nobs (25-dim)", fs=8)

# 점점점
label(6.4, 2.95, "⋮", fs=14, color="#E65100")

# 슬롯 사이 아래 화살표
for sy in [6.4, 5.45, 4.5]:
    arrow(6.4, sy, 6.4, sy - 0.65, color="#FB8C00", txt="next agent")

# LSTM 상태 흐름 (왼쪽 세로 화살표)
ax.annotate("", xy=(4.45, 3.5), xytext=(4.45, 6.4 + SH),
            arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.8,
                            connectionstyle="arc3,rad=-0.4"))
label(4.0, 5.1, "LSTM\nstate\n(h, c)", fs=7.5, color="#7B1FA2")

# 환경 → 분해 화살표
arrow(3.9, 4.7, 4.1, 4.7, color="#1565C0", lw=2, txt="shared\nstate")

# ════════════════════════════════════════════════
# (C) Policy Network  x: 9.2~13.8
# ════════════════════════════════════════════════
box(9.1, 1.2, 4.8, 7.0, C_FC, ec="#2E7D32", lw=2)
label(11.5, 8.0, "RecurrentPPO  (MlpLstmPolicy)", fs=10, fw="bold", color="#1B5E20")

# 입력 obs
box(10.0, 6.5, 3.0, 0.65, C_OBS, ec="#F9A825", lw=1.5)
label(11.5, 6.83, "Observation  (25-dim)\ncell(8) + global(15) + pos(2)", fs=8)

# FC layers
box(10.0, 5.4, 3.0, 0.65, C_FC, ec="#388E3C", lw=1.5)
label(11.5, 5.73, "FC(256)  + ReLU", fs=9)

box(10.0, 4.3, 3.0, 0.65, C_FC, ec="#388E3C", lw=1.5)
label(11.5, 4.63, "FC(256)  + ReLU", fs=9)

# LSTM
box(10.0, 3.1, 3.0, 0.78, C_LSTM, ec="#6A1B9A", lw=1.8)
label(11.5, 3.49, "LSTM  (hidden = 256)", fs=9.5, fw="bold", color="#4A148C")

# Actor / Critic 분기
box(9.6,  1.85, 1.5, 0.7, C_ACTOR,  ec="#C62828", lw=1.5)
label(10.35, 2.2, "Actor\nhead", fs=8.5)

box(11.55, 1.85, 1.5, 0.7, C_CRITIC, ec="#00838F", lw=1.5)
label(12.3, 2.2, "Critic\nhead", fs=8.5)

# 내부 화살표
arrow(11.5, 6.5,  11.5, 6.05, lw=1.3)
arrow(11.5, 5.4,  11.5, 4.95, lw=1.3)
arrow(11.5, 4.3,  11.5, 3.88, lw=1.3)
arrow(11.5, 3.1,  11.5, 2.55, lw=1.3)
arrow(11.5, 2.55, 10.35, 2.55, lw=1.3)
arrow(11.5, 2.55, 12.3,  2.55, lw=1.3)

# Autoregressive → Policy 화살표
arrow(8.7, 5.0, 9.1, 6.7, color="#E65100", lw=2, txt="obs₍ᵢ₎")

# ════════════════════════════════════════════════
# (D) 출력 (Action / Value)  x: 14.3~17.3
# ════════════════════════════════════════════════
# Actor 출력
box(14.3, 4.5, 3.0, 1.4, C_ACTOR, ec="#C62828", lw=1.5)
label(15.8, 5.2, "Action\nDiscrete(2)", fs=9, fw="bold", color="#B71C1C")
label(15.8, 4.8, "Exit A  /  Exit B", fs=8.5, color="#C62828")

# Critic 출력
box(14.3, 2.7, 3.0, 1.0, C_CRITIC, ec="#00838F", lw=1.5)
label(15.8, 3.2, "Value  V(s)", fs=9.5, fw="bold", color="#006064")

# 화살표
arrow(11.1, 2.2, 14.3, 4.95, color="#C62828", lw=1.8, txt="π(a|obs)")
arrow(12.8, 2.2, 14.3, 3.1,  color="#006064", lw=1.8, txt="V")

# 출구 유도 신호
box(14.3, 1.3, 3.0, 1.0, C_EXIT, ec="#E65100", lw=2)
label(15.8, 1.8, "Exit Guidance Signal\n(applied to environment)", fs=8.5, fw="bold", color="#BF360C")
arrow(15.8, 4.5, 15.8, 2.3, color="#E65100", lw=1.8)

# ════════════════════════════════════════════════
# (E) 커리큘럼 학습 하단 바
# ════════════════════════════════════════════════
box(0.3, 0.08, 17.4, 0.85, "#ECEFF1", ec=C_BORDER, lw=1.2)
label(1.35, 0.51, "Curriculum:", fs=8.5, fw="bold", color="#37474F")

stages = [
    ("S1  Basic\n20 agents",  "#A5D6A7", "#2E7D32"),
    ("S2  Exit A\nThreat  40", "#FFF176", "#F9A825"),
    ("S3  Path\nBlock  40",  "#FFCC80", "#E65100"),
    ("S4  Dual\nThreat  40", "#EF9A9A", "#C62828"),
]
for i, (txt, fc, ec) in enumerate(stages):
    sx = 2.8 + i * 3.6
    box(sx, 0.12, 3.0, 0.73, fc, ec=ec, lw=1.2)
    label(sx + 1.5, 0.485, txt, fs=7.5, color=ec)
    if i < 3:
        arrow(sx + 3.0, 0.485, sx + 3.15, 0.485, color="#546E7A", lw=2)

label(6.4, 0.95, f"threshold ≥ 80%  |  window = 50 episodes",
      fs=7.5, color="#546E7A", fw="normal")

# ════════════════════════════════════════════════
# 제목
# ════════════════════════════════════════════════
ax.text(9.0, 8.72,
        "AutoregressivePPO Architecture for Fire Evacuation Guidance",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#212121",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1",
                  edgecolor=C_BORDER, lw=1.5))

plt.tight_layout(pad=0.3)
out = "autoregressive_ppo_architecture.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
