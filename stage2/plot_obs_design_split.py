"""
관측 공간 설계 비교 — 표 + 그래프 분리 출력
  Figure 1: 비교 표
  Figure 2: 수평 막대 그래프 (x·y축 교환, 글씨 2배)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np

_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM):
    font_manager.fontManager.addfont(_NANUM)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_NANUM).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "result", "obs_design")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE  = "#3A7DD4"
RED   = "#D94F3D"
GRAY1 = "#F5F7FA"   # 짝수 행 배경
GRAY2 = "#FFFFFF"   # 홀수 행 배경
HEAD  = "#2C3E50"   # 헤더 배경

# ══════════════════════════════════════════════════════════════
# Figure 1 — 비교 표
# ══════════════════════════════════════════════════════════════
BASE_FS = 18  # 기준 폰트 크기

rows = [
    ("관측 공간 크기",      "15",      "3,000+"),
    ("전체 MLP 파라미터",   "5,379",   "196,419"),
    ("첫 레이어 가중치 수", "1,024",   "192,064"),
    ("학습 성능 (생존율)",  "85 %",    "71 %"),
    ("그리드 정보 필요",    "불필요",   "필요"),
    ("학습 수렴 속도",      "빠름",    "느림"),
]

# 특별 강조 행 인덱스: {row_idx: (badge_text, badge_color)}
HIGHLIGHT_ROWS = {
    3: ("+14%p ▲", "#1B7F3A"),   # 학습 성능 (생존율) — 향상 배지
    5: ("수렴 빠름 ▲", "#1B7F3A"),  # 학습 수렴 속도
}

col_labels = ["항목", "15차원 스칼라\n(제안된 모델)", "3,000차원 그리드\n(비교 대상)"]

fig1, ax1 = plt.subplots(figsize=(13, 5.5))
ax1.axis("off")

col_w = [0.38, 0.28, 0.28]
badge_w = 0.06  # 배지 열 너비
x_pos = [0.0, col_w[0], col_w[0] + col_w[1]]
x_badge = col_w[0] + col_w[1] + col_w[2]  # 배지 시작 x

y_top  = 0.88
row_h  = 0.13

# 헤더
for j, (label, xp, cw) in enumerate(zip(col_labels, x_pos, col_w)):
    rect = mpatches.FancyBboxPatch(
        (xp + 0.005, y_top), cw - 0.01, row_h,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=HEAD, transform=ax1.transAxes, clip_on=False
    )
    ax1.add_patch(rect)
    ax1.text(xp + cw / 2, y_top + row_h / 2, label,
             transform=ax1.transAxes,
             ha="center", va="center",
             fontsize=BASE_FS - 1, fontweight="bold", color="white")

# 데이터 행
for i, (item, v_scalar, v_grid) in enumerate(rows):
    y = y_top - (i + 1) * row_h
    bg = GRAY1 if i % 2 == 0 else GRAY2

    # 강조 행은 배경 연두 처리
    if i in HIGHLIGHT_ROWS:
        bg = "#EAF7ED"

    for j, (val, xp, cw) in enumerate(zip([item, v_scalar, v_grid], x_pos, col_w)):
        rect = mpatches.FancyBboxPatch(
            (xp + 0.005, y), cw - 0.01, row_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bg, transform=ax1.transAxes, clip_on=False
        )
        ax1.add_patch(rect)

        # 색상 결정
        if j == 0:
            color = "#1a1a1a"
            fw = "bold"
        elif j == 1:
            color = BLUE
            fw = "bold"
        else:
            color = RED
            fw = "bold"

        ax1.text(xp + cw / 2, y + row_h / 2, val,
                 transform=ax1.transAxes,
                 ha="center", va="center",
                 fontsize=BASE_FS, fontweight=fw, color=color)

    # 강조 배지 (행 오른쪽 끝)
    if i in HIGHLIGHT_ROWS:
        badge_text, badge_color = HIGHLIGHT_ROWS[i]
        badge_x = x_badge + 0.01
        badge_rect = mpatches.FancyBboxPatch(
            (badge_x, y + 0.015), badge_w + 0.05, row_h - 0.03,
            boxstyle="round,pad=0.01", linewidth=1.5,
            facecolor=badge_color, edgecolor=badge_color,
            transform=ax1.transAxes, clip_on=False
        )
        ax1.add_patch(badge_rect)
        ax1.text(badge_x + (badge_w + 0.05) / 2, y + row_h / 2,
                 badge_text,
                 transform=ax1.transAxes,
                 ha="center", va="center",
                 fontsize=BASE_FS - 2, fontweight="bold", color="white")

# 외곽선
border = mpatches.FancyBboxPatch(
    (0.005, y_top - len(rows) * row_h),
    sum(col_w) - 0.01, (len(rows) + 1) * row_h,
    boxstyle="square,pad=0", linewidth=1.5,
    edgecolor="#AAAAAA", facecolor="none",
    transform=ax1.transAxes, clip_on=False
)
ax1.add_patch(border)

# 하단 요약 callout
summary_y = y_top - (len(rows) + 0.3) * row_h - 0.06
ax1.add_patch(mpatches.FancyBboxPatch(
    (0.005, summary_y - 0.085), sum(col_w) - 0.01, 0.10,
    boxstyle="round,pad=0.01", linewidth=1.5,
    facecolor="#EAF7ED", edgecolor="#1B7F3A",
    transform=ax1.transAxes, clip_on=False
))
ax1.text(sum(col_w) / 2, summary_y - 0.035,
         "200× 차원 축소  →  생존율 +14%p 향상  ·  학습 수렴 속도 ↑",
         transform=ax1.transAxes,
         ha="center", va="center",
         fontsize=BASE_FS, fontweight="bold", color="#1B7F3A")

fig1.suptitle("관측 공간 설계 비교", fontsize=BASE_FS + 4, fontweight="bold", y=1.02)
fig1.tight_layout(rect=[0, 0, 1, 1])
out1 = os.path.join(OUT_DIR, "fig1_table.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"저장: {out1}")


# ══════════════════════════════════════════════════════════════
# Figure 2 — 가로형, 우측 callout, 제목 없음
# ══════════════════════════════════════════════════════════════
import math as _math

BAR_FS     = int(BASE_FS * 2.2)   # 41pt
GREEN_DARK = "#1B7F3A"
GREEN_BG   = "#EAF7ED"

GROUPS = [
    ("관측 가중치 (첫 레이어)",  1_024,  192_064, 188),
    ("MLP 전체 파라미터",        5_379,  196_419,  37),
]
bar_h = 0.55

Y_POS = [(4.1, 2.9), (1.9, 0.7)]
YLIM  = (0.1, 5.4)

def _ay(y):   # data y → transAxes y
    return (y - YLIM[0]) / (YLIM[1] - YLIM[0])

fig2, ax2 = plt.subplots(figsize=(14, 9), dpi=150)
fig2.subplots_adjust(left=0.24, right=0.50, top=0.93, bottom=0.16)

for (title, v_ours, v_grid, ratio), (yg, yo) in zip(GROUPS, Y_POS):
    yc = (yg + yo) / 2

    # ─ ghost bar: 전체 범위 대비 차이 강조 ─
    ax2.barh(yg, 450_000, height=bar_h, color=RED, alpha=0.09, zorder=1)

    # ─ 3,000차원 막대 (RED, 위) — 값 레이블 막대 내부 흰 글씨 ─
    ax2.barh(yg, v_grid, height=bar_h, color=RED, zorder=3)
    ax2.text(v_grid * 0.55, yg, f"{v_grid:,}",
             va="center", ha="center", fontsize=BAR_FS - 4, fontweight="bold",
             color="white", zorder=4)
    ax2.text(-0.03, _ay(yg), "3,000차원\n(비교 대상)",
             ha="right", va="center", fontsize=BAR_FS - 6,
             fontweight="bold", color=RED,
             transform=ax2.transAxes, clip_on=False)

    # ─ 제안된 모델 막대 (BLUE, 아래) — 값 레이블 막대 오른쪽 ─
    ax2.barh(yo, v_ours, height=bar_h, color=BLUE, zorder=3)
    ax2.text(v_ours * 3.5, yo, f"{v_ours:,}",
             va="center", ha="left", fontsize=BAR_FS - 4, fontweight="bold",
             color=BLUE)
    ax2.text(-0.03, _ay(yo), "제안된\n모델",
             ha="right", va="center", fontsize=BAR_FS - 6,
             fontweight="bold", color=BLUE,
             transform=ax2.transAxes, clip_on=False)

    # ─ 배율 어노테이션 — 기하평균 x ─
    x_mid = _math.exp((_math.log(v_ours) + _math.log(v_grid)) / 2)
    ax2.text(x_mid, yc, f"{ratio}×\n더 큼",
             fontsize=BAR_FS, color=RED, fontweight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=RED, alpha=0.90))

    # ─ 그룹 제목: 3,000차원 막대 위에 ─
    ax2.text(600, yg + bar_h / 2 + 0.10, title,
             va="bottom", ha="left", fontsize=BAR_FS - 9,
             fontweight="bold", color="#555555")

# 그룹 구분선
ax2.axhline(2.40, color="#CCCCCC", lw=2.5, ls="--", alpha=0.7, zorder=0)

ax2.set_xscale("log")
ax2.set_xlim(500, 450_000)
ax2.set_ylim(*YLIM)
ax2.set_yticks([])
ax2.set_xlabel("파라미터 수 (로그 스케일)", fontsize=BAR_FS, labelpad=10)
ax2.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)
ax2.tick_params(axis="x", labelsize=BAR_FS - 6, rotation=20)
ax2.grid(axis="x", which="both", alpha=0.20, zorder=0)
ax2.spines[["top", "right", "left"]].set_visible(False)

# ── Callout: 우측 배치 ─────────────────────────
callout_ax = fig2.add_axes([0.54, 0.22, 0.43, 0.56])
callout_ax.axis("off")
callout_ax.add_patch(mpatches.FancyBboxPatch(
    (0.04, 0.04), 0.92, 0.92,
    boxstyle="round,pad=0.04", linewidth=3,
    facecolor=GREEN_BG, edgecolor=GREEN_DARK,
    transform=callout_ax.transAxes, clip_on=False
))
callout_ax.text(0.5, 0.76, "200× 차원 축소",
    transform=callout_ax.transAxes,
    ha="center", va="center",
    fontsize=BAR_FS + 6, fontweight="bold", color=GREEN_DARK)
callout_ax.text(0.5, 0.52, "→  생존율 +14%p 향상",
    transform=callout_ax.transAxes,
    ha="center", va="center",
    fontsize=BAR_FS, fontweight="bold", color=GREEN_DARK)
callout_ax.text(0.5, 0.30, "학습 수렴 속도 ↑",
    transform=callout_ax.transAxes,
    ha="center", va="center",
    fontsize=BAR_FS, fontweight="bold", color=GREEN_DARK)

out2 = os.path.join(OUT_DIR, "fig2_bars.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"저장: {out2}")
