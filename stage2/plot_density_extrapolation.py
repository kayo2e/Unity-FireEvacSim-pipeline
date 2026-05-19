"""
인원 밀도 외삽 실험: 생존율 + 추론 시간 이중 Y축 그래프
====================================================
X축: 인원 수 (20 / 40 / 60 / 80 / 100 / 120 / 140 / 160 / 180 / 200)
좌 Y축: 생존율 (%) — PPO 파란 실선 / A* 빨간 실선
우 Y축: 추론 시간 (ms) — A* 빨간 점선 (선형 증가) / PPO 파란 수평 점선 (상수)

실행:
    python3 plot_density_extrapolation.py --scenario 4 --episodes 30
    python3 plot_density_extrapolation.py --scenario 4 --episodes 10  # 빠른 테스트

저장 위치: result/density_extrapolation/density_s{N}_ep{E}.png
"""

import time
import os
import sys
import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from baselines.astar_real import astar_action

_ROOT     = os.path.dirname(os.path.abspath(__file__))
TRAIN_N   = 40                                               # PPO 학습 인원수
N_LIST    = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # 테스트 인원 목록


# ── PPO 로드 ─────────────────────────────────────────
def load_ppo(scenario: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    path = os.path.join(_ROOT, "model", "ppo", "fire_evac_model_40ppl.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PPO 모델 없음: {path}")

    model = PPO.load(path)
    vecnorm_pkl = path.replace(".zip", "_vecnorm.pkl")
    _vn = None
    if os.path.exists(vecnorm_pkl):
        _tmp = DummyVecEnv([lambda: FireEvacEnv(scenario=scenario, n_agents=TRAIN_N)])
        _vn = VecNormalize.load(vecnorm_pkl, _tmp)
        _vn.training    = False
        _vn.norm_reward = False
    print(f"  PPO 로드: {path}")

    def ppo_fn(env):
        raw_obs = env._get_obs()
        obs = _vn.normalize_obs(np.array([raw_obs]))[0] if _vn else raw_obs
        action, _ = model.predict(obs, deterministic=True)
        return action

    return ppo_fn


# ── 에피소드 실행 → (생존율, 추론시간) ──────────────
def run_episodes(scenario: int, n_agents: int, n_ep: int, policy_fn,
                 seeds: list = None) -> tuple[float, float]:
    cfg           = SCENARIO_CONFIGS[scenario]
    survival_list = []
    time_list     = []

    for ep in range(n_ep):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
        env.reset(seed=seeds[ep] if seeds else None)
        ep_times = []

        for _ in range(cfg["max_steps"]):
            t0     = time.perf_counter()
            action = policy_fn(env)
            ep_times.append((time.perf_counter() - t0) * 1000)  # ms

            _, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        survival_list.append(info["survival_rate"] * 100)
        # 첫 호출(warmup) 제외 후 평균
        if len(ep_times) > 1:
            time_list.append(np.mean(ep_times[1:]))
        elif ep_times:
            time_list.append(ep_times[0])
        env.close()

    return float(np.mean(survival_list)), float(np.mean(time_list))


# ── 그래프 생성 ──────────────────────────────────────
PLOT_N = {20, 40, 80, 120, 160, 200}  # 표시할 인원수

def plot_results(results: dict, scenario: int, n_ep: int, out_path: str):
    # 표시할 포인트만 필터링
    idx        = [i for i, n in enumerate(results["n_agents"]) if n in PLOT_N]
    ns         = [results["n_agents"][i]  for i in idx]
    ppo_surv   = [results["ppo_surv"][i]  for i in idx]
    astar_surv = [results["astar_surv"][i] for i in idx]
    ppo_time   = [results["ppo_time"][i]  for i in idx]
    astar_time = [results["astar_time"][i] for i in idx]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=130)

    # ── 추론 시간 ──
    ax.plot(ns, astar_time, "r-^", lw=3.5, ms=13, label="A* 추론 시간")
    ax.plot(ns, ppo_time,   "b-s", lw=3.5, ms=13, label="PPO 추론 시간")
    ax.set_xlabel("인원 수", fontsize=28)
    ax.set_ylabel("추론 시간 (ms)", fontsize=28)
    max_t = max(astar_time) * 1.28
    ax.set_ylim(0, max_t)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}ms"))
    ax.tick_params(axis="both", labelsize=24)
    ax.grid(True, alpha=0.25, zorder=0)

    # ── 배속 어노테이션 (마지막 2개 포인트) ──
    for n, at, pt in zip(ns[-2:], astar_time[-2:], ppo_time[-2:]):
        speedup = at / pt if pt > 0 else 0
        ax.annotate(
            f"{speedup:.0f}×",
            xy=(n, at),
            xytext=(n - 28, at + max_t * 0.08),
            fontsize=24, color="darkred", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.5),
        )

    # ── 학습 밀도 수직선 ──
    ax.axvline(TRAIN_N, color="black", lw=1.5, ls="--", alpha=0.45, zorder=1)
    ax.text(TRAIN_N + 2, max_t * 0.03, f"학습 밀도\n({TRAIN_N}명)",
            fontsize=20, color="#555555", va="bottom")

    # ── 생존율 텍스트 박스 (전체 데이터 기준) ──
    all_ppo  = results["ppo_surv"]
    all_ast  = results["astar_surv"]
    avg_ppo  = float(np.mean(all_ppo))
    avg_ast  = float(np.mean(all_ast))
    avg_diff = avg_ppo - avg_ast
    textbox  = (
        f"생존율 비교 ({n_ep}회 평균)\n"
        f"  PPO : {avg_ppo:.1f}%\n"
        f"  A*   : {avg_ast:.1f}%\n"
        f"  차이 : {avg_diff:+.1f}%p  →  PPO ≈ A*"
    )
    ax.text(
        0.03, 0.97, textbox,
        transform=ax.transAxes,
        fontsize=22, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow",
                  edgecolor="#cccccc", alpha=0.9),
    )

    # ── X축 레이블 ──
    x_labels = []
    for n in ns:
        ratio = n / TRAIN_N
        frac  = f"×{int(ratio)}" if ratio == int(ratio) else f"×{ratio:.1f}"
        x_labels.append(f"{n}명\n({frac})")
    ax.set_xticks(ns)
    ax.set_xticklabels(x_labels, fontsize=22)

    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.68),
              fontsize=24, framealpha=0.88, edgecolor="#cccccc")

    plt.title(
        f"인원 밀도 외삽 실험 — Scenario {scenario}  (PPO 학습: {TRAIN_N}명 | {n_ep}회/인원수)\n"
        f"추론 시간: PPO 일정(~1ms) vs A* 선형 증가",
        fontsize=24,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  그래프 저장: {out_path}")


# ── 결과 표 출력 ──────────────────────────────────────
def print_table(results: dict):
    ns = results["n_agents"]
    header = f"{'인원':>7} {'A* 생존율':>10} {'PPO 생존율':>10} {'PPO 우위':>9} {'A* 추론':>10} {'PPO 추론':>10}"
    print(f"\n{header}")
    print("─" * 65)
    for i, n in enumerate(ns):
        diff    = results["ppo_surv"][i] - results["astar_surv"][i]
        sign    = "+" if diff >= 0 else ""
        marker  = " ←" if n == TRAIN_N else ""
        print(
            f"{n:>6}명{marker:<3} "
            f"{results['astar_surv'][i]:>9.1f}% "
            f"{results['ppo_surv'][i]:>9.1f}% "
            f"{sign}{diff:>8.1f}% "
            f"{results['astar_time'][i]:>9.2f}ms "
            f"{results['ppo_time'][i]:>9.2f}ms"
        )
    print("─" * 65)


# ── 진입점 ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="인원 밀도 외삽 실험")
    parser.add_argument("--scenario", type=int,  default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--episodes", type=int,  default=30, help="인원수당 에피소드 수")
    args = parser.parse_args()

    scenario = args.scenario
    n_ep     = args.episodes

    print(f"\n{'═'*60}")
    print(f"  인원 밀도 외삽 실험  |  Scenario {scenario}  |  {n_ep}회/인원수")
    print(f"  테스트 인원: {N_LIST}")
    print(f"{'═'*60}")

    ppo_fn = load_ppo(scenario)

    results = {
        "n_agents":   N_LIST,
        "astar_surv": [],
        "ppo_surv":   [],
        "astar_time": [],
        "ppo_time":   [],
    }

    seeds = list(range(1000, 1000 + n_ep))  # 고정 시드: 두 정책이 동일한 환경에서 평가

    for n in N_LIST:
        print(f"\n  ── n_agents={n:>3}명 ({n_ep}회) ──")

        astar_surv, astar_time = run_episodes(scenario, n, n_ep, astar_action, seeds=seeds)
        ppo_surv,   ppo_time   = run_episodes(scenario, n, n_ep, ppo_fn,       seeds=seeds)

        results["astar_surv"].append(astar_surv)
        results["ppo_surv"].append(ppo_surv)
        results["astar_time"].append(astar_time)
        results["ppo_time"].append(ppo_time)

        diff = ppo_surv - astar_surv
        print(f"    A*:  생존율 {astar_surv:.1f}%  추론 {astar_time:.2f}ms")
        print(f"    PPO: 생존율 {ppo_surv:.1f}%  추론 {ppo_time:.2f}ms  "
              f"({'PPO +' if diff>=0 else 'A* +'}{abs(diff):.1f}%p)")

    print_table(results)

    out_dir  = os.path.join(_ROOT, "result", "density_extrapolation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"density_s{scenario}_ep{n_ep}.png")
    plot_results(results, scenario, n_ep, out_path)


if __name__ == "__main__":
    main()
