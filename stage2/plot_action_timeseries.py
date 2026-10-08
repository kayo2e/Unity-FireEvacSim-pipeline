"""
예시 에피소드의 cA/cB/wc 시계열 그림 (experiments/exp_action_timeseries.py 출력 사용).

사용법:
  python plot_action_timeseries.py --scenario 4 --episode 29
  python plot_action_timeseries.py --scenario 4 --episode 29 --full-range
"""
import os
import csv
import argparse
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
RESULT_DIR = os.path.join("result", "action_timeseries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=4)
    parser.add_argument("--episode", type=int, default=29)
    parser.add_argument("--full-range", action="store_true")
    args = parser.parse_args()

    path = os.path.join(RESULT_DIR, f"action_timeseries_s{args.scenario}.csv")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if int(r["episode"]) == args.episode]
    col = lambda k: [float(r[k]) for r in rows]
    t = col("step")

    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 0.8, 1]})
    ax = axes[0]
    ax.plot(t, col("cA"), color=BLUE, lw=2, label="cA (EXIT A 비용)")
    ax.plot(t, col("cB"), color=ORANGE, lw=2, label="cB (EXIT B 비용)")
    ax.axhline(27.5, color=GRAY, lw=1, ls="--", label="초기화 중간값 27.5")
    ax.set_ylabel("출구 비용\n(허용 범위 5~50)")
    if args.full_range:
        ax.axhspan(5, 50, color="#f2f2ef", zorder=0)
        ax.set_ylim(0, 55)
        ax.set_yticks([5, 27.5, 50])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=3, fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(t, col("wc"), color=AQUA, lw=2)
    ax.axhline(2.75, color=GRAY, lw=1, ls="--")
    ax.set_ylabel("wc 혼잡 가중치\n(허용 범위 0.5~5)")
    if args.full_range:
        ax.axhspan(0.5, 5, color="#f2f2ef", zorder=0)
        ax.set_ylim(0, 5.5)
        ax.set_yticks([0.5, 2.75, 5])

    ax = axes[2]
    ax.plot(t, col("F1_exitA_threat"), color=BLUE, lw=2, label="F1 EXIT A 안전도")
    ax.plot(t, col("F2_exitB_threat"), color=ORANGE, lw=2, label="F2 EXIT B 안전도")
    ax.set_ylabel("출구 안전도\n(1=안전, 0=위험)")
    ax.set_xlabel("스텝")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=3, fontsize=8, frameon=False)

    for ax in axes:
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    seed = rows[0]["seed"]
    last = rows[-1]
    fig.suptitle(f"S{args.scenario} 예시 에피소드 (seed {seed}) · 탈출 A {last['escaped_A']} / "
                 f"B {last['escaped_B']}, 사망 {last['dead']}", fontsize=11)
    fig.tight_layout()
    suffix = "_fullrange" if args.full_range else ""
    out = os.path.join(RESULT_DIR, f"action_timeseries_s{args.scenario}_ep{args.episode}{suffix}.png")
    fig.savefig(out, dpi=200)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
