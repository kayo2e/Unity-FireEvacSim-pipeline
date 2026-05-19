"""
exp1 summary JSON → 확장 비교 표 PNG
지표: 생존율 / 탈출 소요 시간 / 사망자 수 / 출구 분산(A:B 비율)
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM):
    font_manager.fontManager.addfont(_NANUM)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_NANUM).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

_ROOT    = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.join(_ROOT, "result", "exp1_compare")

# ── 데이터 로드 ──────────────────────────────────────
FILES = {
    1: "exp1_s1_s1_20260514_135043_summary.json",
    2: "exp1_s2_s2_20260514_135337_summary.json",
    3: "exp1_s3_s3_20260514_135613_summary.json",
    4: "exp1_s4_s4_20260515_100328_summary.json",
    6: "exp1_s6_s6_20260515_111234_summary.json",   # 포스터 S5
}
SC_NAME = {
    1: ("S1", "기본 탈출",    "학습 기반"),
    2: ("S2", "EXIT A 위협",  "학습 기반"),
    3: ("S3", "진입로 차단",  "학습 기반"),
    4: ("S4", "양방향 위협",  "학습 기반"),
    6: ("S5", "중앙 차단",    "미학습"),
}

rows = []
for sc, fname in FILES.items():
    with open(os.path.join(_EXP_DIR, fname)) as f:
        d = json.load(f)

    a = d["astar"]
    p = d["ppo"]

    def fmt(mean, std, pct=False):
        scale = 100 if pct else 1
        return f"{mean*scale:.1f}±{std*scale:.1f}"

    def exit_ratio(ea, eb):
        total = ea + eb
        if total == 0:
            return "—"
        return f"{ea/total*100:.0f}:{eb/total*100:.0f}"

    sc_tag, sc_name, cond = SC_NAME[sc]

    surv_a   = a["survival_rate"]["mean"]
    surv_p   = p["survival_rate"]["mean"]
    delta_surv = (surv_p - surv_a) * 100

    step_a   = a["steps_taken"]["mean"]
    step_p   = p["steps_taken"]["mean"]
    delta_step = step_p - step_a

    dead_a   = a["dead"]["mean"]
    dead_p   = p["dead"]["mean"]

    bal_a    = exit_ratio(a["escaped_A"]["mean"], a["escaped_B"]["mean"])
    bal_p    = exit_ratio(p["escaped_A"]["mean"], p["escaped_B"]["mean"])

    rows.append([
        sc_tag, sc_name, cond,
        fmt(surv_a, a["survival_rate"]["std"], pct=True),
        fmt(surv_p, p["survival_rate"]["std"], pct=True),
        f"{delta_surv:+.1f}%p",
        f"{step_a:.0f}±{a['steps_taken']['std']:.0f}",
        f"{step_p:.0f}±{p['steps_taken']['std']:.0f}",
        f"{delta_step:+.0f}",
        f"{dead_a:.1f}",
        f"{dead_p:.1f}",
        bal_a,
        bal_p,
    ])

# ── 표 그리기 ─────────────────────────────────────────
COL_LABELS = [
    "시나리오", "조건", "학습 조건",
    "A* 생존율(%)", "PPO 생존율(%)", "Δ생존율",
    "A* step", "PPO step", "Δstep",
    "A* 사망자", "PPO 사망자",
    "A* 출구(A:B)", "PPO 출구(A:B)",
]

COL_W = [0.055, 0.09, 0.075,
         0.095, 0.095, 0.065,
         0.085, 0.085, 0.055,
         0.07,  0.07,
         0.075, 0.075]

fig, ax = plt.subplots(figsize=(22, 4.2), dpi=130)
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=COL_LABELS,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)

# 열 너비 지정
for ci, w in enumerate(COL_W):
    for ri in range(len(rows) + 1):
        tbl[ri, ci].set_width(w)

# 헤더 스타일
HEADER_COLOR = "#2E4A7A"
for ci in range(len(COL_LABELS)):
    cell = tbl[0, ci]
    cell.set_facecolor(HEADER_COLOR)
    cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    cell.set_height(0.22)

# 그룹 헤더 색: 생존율(파랑), step(녹색), 사망자(빨강), 출구분산(보라)
GROUP_COLORS = {
    (3, 4, 5): "#dce6f1",   # 생존율
    (6, 7, 8): "#e2efda",   # step
    (9, 10):   "#fce4d6",   # 사망자
    (11, 12):  "#ede7f6",   # 출구분산
}

# 행 스타일
for ri, row in enumerate(rows, start=1):
    sc_tag = row[0]
    base_c = "#f9f9f9" if ri % 2 == 0 else "white"
    for ci in range(len(COL_LABELS)):
        cell = tbl[ri, ci]
        cell.set_height(0.17)

        # 그룹 배경
        gc = base_c
        for cols, color in GROUP_COLORS.items():
            if ci in cols:
                # 짝수/홀수 행 약간 어둡게
                import matplotlib.colors as mc
                rgb = mc.to_rgb(color)
                factor = 0.97 if ri % 2 == 0 else 1.0
                gc = tuple(min(1.0, v * factor) for v in rgb)
        cell.set_facecolor(gc)

        # Δ 컬럼 색상
        text = row[ci - 0] if ci > 0 else ""  # just a placeholder
        val  = row[ci]
        if ci == 5:  # Δ생존율
            try:
                v = float(val.replace("%p", ""))
                if v > 0:
                    cell.set_text_props(color="#1565C0", fontweight="bold")
                elif v < 0:
                    cell.set_text_props(color="#C62828", fontweight="bold")
            except: pass
        elif ci == 8:  # Δstep (음수=PPO가 빠름=좋음)
            try:
                v = float(val)
                if v < 0:
                    cell.set_text_props(color="#1565C0", fontweight="bold")
                elif v > 0:
                    cell.set_text_props(color="#C62828", fontweight="bold")
            except: pass

# 열 그룹 구분선 강조 (생존율·step·사망자·출구분산 구분)
DIVIDER_COLS = [3, 6, 9, 11]
for ri in range(len(rows) + 1):
    for ci in DIVIDER_COLS:
        tbl[ri, ci].set_edgecolor("#888888")

# 제목
fig.text(0.5, 0.97,
         "시나리오별 A* vs PPO 성능 비교  (에피소드 30회 평균)",
         ha="center", va="top",
         fontsize=16, fontweight="bold", color="#1A1A1A")

# 그룹 레이블 (표 위)
ax_pos = ax.get_position()
group_labels = [
    (3, 5,  "생존율",   "#2166ac"),
    (6, 8,  "탈출 소요 시간 (step)", "#33691e"),
    (9, 10, "사망자 수", "#bf360c"),
    (11, 12,"출구 분산 (A:B%)", "#4527a0"),
]
col_positions = []
x_cursor = ax_pos.x0
for ci, w in enumerate(COL_W):
    col_positions.append(x_cursor + w / 2)
    x_cursor += w

for c_start, c_end, label, color in group_labels:
    x_mid = (col_positions[c_start] + col_positions[c_end]) / 2
    fig.text(x_mid, ax_pos.y1 + 0.015, label,
             ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=color,
             transform=fig.transFigure)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = os.path.join(_ROOT, "result", "density_extrapolation", "exp1_table_extended.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"저장 완료: {out_path}")