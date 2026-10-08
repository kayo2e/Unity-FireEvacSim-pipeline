"""
S2와 S5 예시 에피소드의 출구 비용(cA, cB)과 출구 안전도(F1, F2) 비교 그림
(experiments/exp_action_timeseries.py 출력 사용).

사용법:
  python plot_action_compare.py
  python plot_action_compare.py --s2-episode 27 --s5-episode 7
"""
import os
import csv
import argparse
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
RESULT_DIR = os.path.join("result", "action_timeseries")
TITLES = {2: "S2 출구 A 위협", 5: "S5 출구 B 위협"}


def load(scenario, episode):
    path = os.path.join(RESULT_DIR, f"action_timeseries_s{scenario}.csv")
    return [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if int(r["episode"]) == episode]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2-episode", type=int, default=27)
    parser.add_argument("--s5-episode", type=int, default=7)
    args = parser.parse_args()

    eps = {2: load(2, args.s2_episode), 5: load(5, args.s5_episode)}
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex="col",
                             gridspec_kw={"height_ratios": [1.2, 1]})

    cost_vals = [float(r[k]) for rows in eps.values() for r in rows for k in ("cA", "cB")]
    lo, hi = min(cost_vals) - 0.2, max(cost_vals) + 0.2

    for j, (sc, rows) in enumerate(eps.items()):
        col = lambda k: [float(r[k]) for r in rows]
        t = col("step")
        last = rows[-1]

        ax = axes[0, j]
        ax.plot(t, col("cA"), color=BLUE, lw=2, label="cA (출구 A 비용)")
        ax.plot(t, col("cB"), color=ORANGE, lw=2, label="cB (출구 B 비용)")
        ax.axhline(27.5, color=GRAY, lw=1, ls="--", label="초기화값 27.5")
        ax.set_ylim(lo, hi)
        ax.set_title(f"{TITLES[sc]} (seed {rows[0]['seed']}) · 탈출 A {last['escaped_A']} / "
                     f"B {last['escaped_B']}, 사망 {last['dead']}", fontsize=10)
        if j == 0:
            ax.set_ylabel("출구 비용\n(허용 범위 5~50)")

        ax = axes[1, j]
        ax.plot(t, col("F1_exitA_threat"), color=BLUE, lw=2, label="F1 출구 A 안전도")
        ax.plot(t, col("F2_exitB_threat"), color=ORANGE, lw=2, label="F2 출구 B 안전도")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("스텝")
        if j == 0:
            ax.set_ylabel("출구 안전도\n(1=안전, 0=위험)")

    for ax in axes.flat:
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = os.path.join(RESULT_DIR,
                       f"action_compare_s2ep{args.s2_episode}_s5ep{args.s5_episode}.png")
    fig.savefig(out, dpi=200)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
