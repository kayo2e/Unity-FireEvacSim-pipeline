"""
관측 공간 비교: 15차원 스칼라 피처 vs 3,000차원 그리드 플래튼
=================================================================
비교 지표:
  - 설계 특성 비교표 (정성 + 정량 혼합)
  - 관측층 파라미터 수 (첫 번째 Linear 레이어 가중치)
  - 전체 네트워크 파라미터 수

저장: result/obs_comparison/obs_comparison.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager

_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM):
    font_manager.fontManager.addfont(_NANUM)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_NANUM).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── 시나리오별 실측 데이터: exp1_compare JSON에서 동적 로드 ──
import json, glob

def _load_scenario_data():
    exp_dir = os.path.join(_ROOT, "result", "exp1_compare")
    sc_info = {
        1: {"label": "S1\n(기본)"},
        2: {"label": "S2\n(EXIT A 위협)"},
        3: {"label": "S3\n(진입로 차단)"},
        4: {"label": "S4\n(양방향 위협)"},
    }
    for sc in [1, 2, 3, 4]:
        # 가장 최근 파일 사용
        pattern = os.path.join(exp_dir, f"exp1_s{sc}_s{sc}_*_summary.json")
        files   = sorted(glob.glob(pattern))
        if not files:
            sc_info[sc].update({"astar": 0, "ppo": 0, "astar_std": 0, "ppo_std": 0})
            continue
        with open(files[-1]) as f:
            d = json.load(f)
        sc_info[sc]["astar"]     = d["astar"]["survival_rate"]["mean"] * 100
        sc_info[sc]["ppo"]       = d["ppo"]["survival_rate"]["mean"]   * 100
        sc_info[sc]["astar_std"] = d["astar"]["survival_rate"]["std"]  * 100
        sc_info[sc]["ppo_std"]   = d["ppo"]["survival_rate"]["std"]    * 100
    return [sc_info[s] for s in [1, 2, 3, 4]]

SCENARIO_DATA = _load_scenario_data()

# ── 설계 상수 ────────────────────────────────────────────
OBS_15   = 15
OBS_3K   = 3_000   # fire_map + smoke_map + people_map (40×25 × 3채널 = 3,000)
HIDDEN   = 256      # 실제 학습에 사용된 hidden size (net_arch=[256,256])
ACTION   = 3

# 실제 PPO 파라미터: 학습된 모델에서 직접 로드
def _load_ppo_params():
    from stable_baselines3 import PPO
    path = os.path.join(_ROOT, "model", "ppo", "fire_evac_model_40ppl.zip")
    m = PPO.load(path)
    return sum(p.numel() for p in m.policy.parameters())

PPO_REAL_PARAMS = _load_ppo_params()   # 실측: 140,807

# 3000차원 모델 파라미터 (동일 net_arch=[256,256])
# actor+critic 각각: (3000×256+256) + (256×256+256) → ×2 + action/value head
LAYER1_3K_ACTOR = OBS_3K * HIDDEN + HIDDEN
MLP_TOTAL_3K    = (
    (OBS_3K * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN) * 2  # actor + critic mlp
    + HIDDEN * ACTION + ACTION                                   # action head
    + HIDDEN * 1     + 1                                         # value head
)

# 15차원 동일 구조 이론값 (참고용)
LAYER1_15_ACTOR = OBS_15 * HIDDEN + HIDDEN
MLP_TOTAL_15    = (
    (OBS_15 * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN) * 2
    + HIDDEN * ACTION + ACTION
    + HIDDEN * 1     + 1
)


# ── 그래프 ────────────────────────────────────────────────
def plot_comparison(out_path: str):
    labels = ["15차원\n스칼라 피처", "3,000차원\n그리드 플래튼"]
    colors = ["#1E90FF", "#FF6B6B"]

    fig = plt.figure(figsize=(13, 10), dpi=130)
    # 2행 3열: 상단=설계비교, 하단=시나리오 성능
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           width_ratios=[1.75, 1, 1],
                           height_ratios=[1.1, 1],
                           hspace=0.45, wspace=0.38)
    ax_tbl   = fig.add_subplot(gs[0, 0])
    ax_lay1  = fig.add_subplot(gs[0, 1])
    ax_total = fig.add_subplot(gs[0, 2])
    ax_sc    = fig.add_subplot(gs[1, :])

    # ────── 비교 표 ────────────────────────────────────────
    ax_tbl.axis("off")
    col_labels = ["항목", "15차원 스칼라", "3,000차원 그리드"]

    def chk(ok):
        return "O (가능)" if ok else "X (불가)"

    s3_ppo  = next(d["ppo"]  for d in SCENARIO_DATA if "S3" in d["label"])
    s3_astr = next(d["astar"] for d in SCENARIO_DATA if "S3" in d["label"])
    rows = [
        ["관측 차원",          f"{OBS_15}",               f"{OBS_3K:,}"],
        ["네트워크 구조",      "net_arch=[256,256]",       "net_arch=[256,256]"],
        ["관측층 가중치",      f"{LAYER1_15_ACTOR:,}",     f"{LAYER1_3K_ACTOR:,}"],
        ["전체 파라미터",      f"{PPO_REAL_PARAMS:,}",     f"{MLP_TOTAL_3K:,}"],
        ["생존율 (S3, 40명)",  f"{s3_ppo:.1f}%",           "— (미학습)"],
        ["에이전트 수 독립",   chk(True),                  chk(False)],
        ["그리드 크기 독립",   chk(True),                  chk(False)],
        ["Unity 이식",         "O (15값 전송)",             "X (40×25×3 전송)"],
        ["학습 수렴 속도",     "빠름",                      "느림"],
    ]

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.55)

    # 헤더
    for j in range(3):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # 15차원: 연파랑, 3000차원: 연빨강
    for i in range(1, len(rows) + 1):
        tbl[i, 1].set_facecolor("#EAF4FF")
        tbl[i, 2].set_facecolor("#FFF0F0")

    ax_tbl.set_title("관측 공간 설계 비교", fontsize=13,
                     fontweight="bold", pad=10)

    # ────── 관측층 파라미터 막대 ──────────────────────────
    vals1 = [LAYER1_15_ACTOR, LAYER1_3K_ACTOR]
    bars = ax_lay1.bar(labels, vals1, color=colors, width=0.5,
                       edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals1):
        ax_lay1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(vals1) * 0.02,
                     f"{v:,}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    ratio1 = vals1[1] / vals1[0]
    ax_lay1.annotate(
        f"{ratio1:.0f}x\n더 많음",
        xy=(1, vals1[1]), xytext=(0.55, vals1[1] * 0.6),
        fontsize=12, color="#B71C1C", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_lay1.set_ylabel("파라미터 수", fontsize=12)
    ax_lay1.set_title("관측층 가중치 수\n(첫 번째 Linear, actor)",
                      fontsize=11, fontweight="bold")
    ax_lay1.set_ylim(0, max(vals1) * 1.3)
    ax_lay1.tick_params(axis="x", labelsize=10)
    ax_lay1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x)))
    )
    ax_lay1.grid(axis="y", alpha=0.3)

    # ────── 전체 파라미터 막대 ────────────────────────────
    vals2 = [MLP_TOTAL_15, MLP_TOTAL_3K]
    bars2 = ax_total.bar(labels, vals2, color=colors, width=0.5,
                         edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars2, vals2):
        ax_total.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(vals2) * 0.02,
                      f"{v:,}", ha="center", va="bottom",
                      fontsize=10, fontweight="bold")

    ratio2 = MLP_TOTAL_3K / MLP_TOTAL_15
    ax_total.annotate(
        f"{ratio2:.0f}x\n더 많음",
        xy=(1, MLP_TOTAL_3K), xytext=(0.55, MLP_TOTAL_3K * 0.6),
        fontsize=12, color="#B71C1C", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_total.set_ylabel("파라미터 수", fontsize=12)
    ax_total.set_title("MLP 전체 파라미터\n(동일 hidden=[64,64] 구조)", fontsize=11,
                       fontweight="bold")
    ax_total.set_ylim(0, max(vals2) * 1.3)
    ax_total.tick_params(axis="x", labelsize=10)
    ax_total.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x)))
    )
    ax_total.grid(axis="y", alpha=0.3)

    # ────── 시나리오별 생존율 막대그래프 ───────────────────
    sc_labels  = [d["label"]     for d in SCENARIO_DATA]
    astar_surv = [d["astar"]     for d in SCENARIO_DATA]
    ppo_surv   = [d["ppo"]       for d in SCENARIO_DATA]
    astar_std  = [d["astar_std"] for d in SCENARIO_DATA]
    ppo_std    = [d["ppo_std"]   for d in SCENARIO_DATA]

    x     = np.arange(len(sc_labels))
    width = 0.32
    err_kw = dict(capsize=4, capthick=1.2, elinewidth=1.2, ecolor="#555")

    bars_a = ax_sc.bar(x - width / 2, astar_surv, width,
                       color="#FF6B6B", label="A*",
                       yerr=astar_std, error_kw=err_kw,
                       edgecolor="white", linewidth=1.0)
    bars_p = ax_sc.bar(x + width / 2, ppo_surv, width,
                       color="#1E90FF", label="PPO (15차원)",
                       yerr=ppo_std, error_kw=err_kw,
                       edgecolor="white", linewidth=1.0)

    # 수치 레이블
    for bar, val in zip(bars_a, astar_surv):
        ax_sc.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 2.5,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    for bar, val in zip(bars_p, ppo_surv):
        ax_sc.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 2.5,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # PPO 우위 어노테이션
    for i, d in enumerate(SCENARIO_DATA):
        diff = d["ppo"] - d["astar"]
        sign = "+" if diff >= 0 else ""
        clr  = "#1565C0" if diff >= 0 else "#B71C1C"
        ax_sc.text(i, max(d["astar"], d["ppo"]) + max(astar_std[i], ppo_std[i]) + 8,
                   f"PPO {sign}{diff:.1f}%p",
                   ha="center", va="bottom", fontsize=9.5,
                   color=clr, fontweight="bold")

    ax_sc.set_xticks(x)
    ax_sc.set_xticklabels(sc_labels, fontsize=12)
    ax_sc.set_ylabel("생존율 (%)", fontsize=12)
    ax_sc.set_ylim(0, 130)
    ax_sc.set_title(
        "시나리오별 생존율 비교 — A* vs PPO (n_agents=40, 30회 평균 ± std)",
        fontsize=12, fontweight="bold", pad=8)
    ax_sc.legend(fontsize=11, loc="upper right", framealpha=0.88)
    ax_sc.grid(axis="y", alpha=0.3)
    ax_sc.axhline(100, color="#888", lw=0.8, ls="--", alpha=0.5)

    fig.suptitle(
        "관측 공간 설계 비교: 15차원 스칼라 피처 vs 3,000차원 그리드 플래튼\n"
        f"(fire_map + smoke_map + people_map  ×  40×25 = {OBS_3K:,}차원 | MLP hidden=[64,64])",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  저장: {out_path}")


# ── 결과 표 출력 ─────────────────────────────────────────
def print_table():
    ratio_l1 = LAYER1_3K / LAYER1_15
    ratio_tot = MLP_TOTAL_3K / MLP_TOTAL_15
    print(f"\n{'─'*60}")
    print(f"  {'항목':<22} {'15차원':>12} {'3,000차원':>14}")
    print(f"{'─'*60}")
    print(f"  {'관측 차원':<22} {OBS_15:>12,} {OBS_3K:>14,}")
    print(f"  {'관측층 가중치 수':<22} {LAYER1_15:>12,} {LAYER1_3K:>14,}  ({ratio_l1:.0f}×)")
    print(f"  {'MLP 전체 파라미터':<22} {MLP_TOTAL_15:>12,} {MLP_TOTAL_3K:>14,}  ({ratio_tot:.0f}×)")
    print(f"  {'생존율 (S3, 40명)':<22} {'84.0%':>12} {'— (미학습)':>14}")
    print(f"  {'에이전트 수 독립':<22} {'O':>12} {'X':>14}")
    print(f"  {'그리드 크기 독립':<22} {'O':>12} {'X':>14}")
    print(f"  {'Unity 이식 가능성':<22} {'O':>12} {'X':>14}")
    print(f"{'─'*60}")


# ── 진입점 ────────────────────────────────────────────────
def main():
    print_table()
    out = os.path.join(_ROOT, "result", "obs_comparison", "obs_comparison.png")
    plot_comparison(out)
    print("완료.")


if __name__ == "__main__":
    main()
