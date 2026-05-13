"""
exp1_compare.py — 실험 ①: A* 베이스라인 vs PPO 직접 비교
==========================================================
4개 시나리오 전체에서 생존율 / 탈출시간 / 출구분배 / 공황지수 비교.

실행:
    cd stage2
    python experiments/exp1_compare.py
    python experiments/exp1_compare.py --episodes 50
    python experiments/exp1_compare.py --model-dir model/recurrent_ppo --model-cls recurrent
"""

import sys, os, json, csv, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from astar_baseline import rule_based_action

RESULT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "result", "exp1_compare")


# ── A* 테스트 ─────────────────────────────────
def run_astar(scenario: int, n_agents: int, n_episodes: int) -> list:
    cfg = SCENARIO_CONFIGS[scenario]
    records = []
    for ep in range(n_episodes):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
        obs, _ = env.reset()
        total_r = 0.0
        info = {}
        for _ in range(cfg["max_steps"]):
            action = rule_based_action(obs)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                break
        env.close()
        records.append(_make_rec(ep + 1, scenario, n_agents, info, total_r))
        _print_ep(ep + 1, n_episodes, records[-1])
    return records


# ── PPO 테스트 ────────────────────────────────
def run_ppo(scenario: int, n_agents: int, n_episodes: int,
            model_dir: str, model_cls_name: str) -> list:
    cfg = SCENARIO_CONFIGS[scenario]

    ModelCls = _load_model_cls(model_cls_name)
    model_n, model_path, vecnorm_path = _find_model(model_dir, n_agents)
    if model_path is None:
        print(f"  [경고] {model_dir} 에 모델 없음 — PPO 테스트 건너뜀")
        return []

    print(f"  모델 로드: {model_path} (학습 인원 {model_n}명 → 테스트 {n_agents}명)")
    model = ModelCls.load(model_path)

    records = []
    for ep in range(n_episodes):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
        vec = DummyVecEnv([lambda: env])
        if os.path.exists(vecnorm_path):
            vec = VecNormalize.load(vecnorm_path, vec)
            vec.training = False
            vec.norm_reward = False

        obs = vec.reset()
        lstm_states = None
        ep_starts = np.ones((1,), dtype=bool)
        total_r = 0.0
        info = {}

        for _ in range(cfg["max_steps"]):
            if model_cls_name == "recurrent":
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
                ep_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = vec.step(action)
            total_r += float(r[0])
            if infos[0]:
                info = infos[0]
            if done[0]:
                break
        vec.close()

        records.append(_make_rec(ep + 1, scenario, n_agents, info, total_r))
        _print_ep(ep + 1, n_episodes, records[-1])
    return records


# ── 유틸 ──────────────────────────────────────
def _make_rec(ep, scenario, n_agents, info, total_r):
    return {
        "episode":       ep,
        "scenario":      scenario,
        "scenario_name": SCENARIO_CONFIGS[scenario]["name"],
        "n_agents":      n_agents,
        "survived":      info.get("escaped", 0),
        "escaped_A":     info.get("escaped_A", 0),
        "escaped_B":     info.get("escaped_B", 0),
        "dead":          info.get("dead", 0),
        "survival_rate": round(info.get("survival_rate", 0.0), 4),
        "steps_taken":   info.get("step", 0),
        "mean_panic":    round(info.get("mean_panic", 0.0), 4),
        "total_reward":  round(total_r, 2),
        "blocked_exits": str(info.get("blocked_exits", [])),
    }


def _print_ep(ep, total, rec):
    print(f"  [ep {ep:>3}/{total}] 생존율 {rec['survival_rate']:.0%} | "
          f"A:{rec['escaped_A']} B:{rec['escaped_B']} | "
          f"공황 {rec['mean_panic']:.2f} | {rec['steps_taken']}스텝")


def _stats(vals):
    a = np.array(vals, dtype=float)
    return {"mean": round(float(a.mean()), 4), "std":  round(float(a.std()),  4),
            "min":  round(float(a.min()),  4), "max":  round(float(a.max()),  4)}


def _load_model_cls(name: str):
    if name == "recurrent":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    from stable_baselines3 import PPO
    return PPO


def _find_model(model_dir: str, n_agents: int):
    """n_agents에 가장 가까운 학습된 모델 파일 탐색."""
    candidates = []
    for fname in os.listdir(model_dir) if os.path.isdir(model_dir) else []:
        if fname.endswith(".zip") and "ppl" in fname:
            try:
                n = int(fname.replace("fire_evac_model_", "").replace("ppl.zip", ""))
                candidates.append(n)
            except ValueError:
                pass
    if not candidates:
        return None, None, None
    best = min(candidates, key=lambda x: abs(x - n_agents))
    path = os.path.join(model_dir, f"fire_evac_model_{best}ppl")
    return best, path, path + "_vecnorm.pkl"


def _print_comparison(sc: int, astar_recs: list, ppo_recs: list, ppo_label: str):
    print(f"\n{'═'*66}")
    print(f"  S{sc} {SCENARIO_CONFIGS[sc]['name']}  "
          f"({SCENARIO_CONFIGS[sc]['n_agents']}명 × {len(astar_recs)}회)")
    print(f"{'─'*66}")
    print(f"  {'지표':<18} {'A* 베이스라인':>14}  {'PPO (' + ppo_label + ')':>16}")
    print(f"{'─'*66}")

    metrics = [("생존율", "survival_rate"), ("탈출(A)", "escaped_A"),
               ("탈출(B)", "escaped_B"), ("사망자", "dead"),
               ("공황지수", "mean_panic"), ("탈출시간", "steps_taken")]
    for label, key in metrics:
        a_vals = [r[key] for r in astar_recs]
        p_vals = [r[key] for r in ppo_recs] if ppo_recs else []
        a_m = np.mean(a_vals)
        p_m = np.mean(p_vals) if p_vals else float("nan")
        diff = p_m - a_m if p_vals else float("nan")
        arrow = ("▲" if diff > 0 else "▼") if not np.isnan(diff) else " "
        if key == "survival_rate":
            print(f"  {label:<18} {a_m:>12.1%}  {p_m:>14.1%}  {arrow}{abs(diff):.1%}")
        else:
            print(f"  {label:<18} {a_m:>12.2f}  {p_m:>14.2f}  {arrow}{abs(diff):.2f}")
    print(f"{'═'*66}")


def _save_results(scenario, astar_recs, ppo_recs, ppo_label):
    os.makedirs(RESULT_BASE, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"s{scenario}_{ts}"
    all_recs = [{"model": "astar", **r} for r in astar_recs] + \
               [{"model": ppo_label, **r} for r in ppo_recs]
    csv_path = os.path.join(RESULT_BASE, f"exp1_s{scenario}_{tag}.csv")
    if all_recs:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_recs[0].keys())
            writer.writeheader()
            writer.writerows(all_recs)
    json_path = os.path.join(RESULT_BASE, f"exp1_s{scenario}_{tag}_summary.json")
    summary = {
        "scenario": scenario, "n_episodes": len(astar_recs),
        "astar":  {k: _stats([r[k] for r in astar_recs])
                   for k in ("survival_rate", "escaped_A", "escaped_B", "dead", "steps_taken")},
        "ppo":    {k: _stats([r[k] for r in ppo_recs])
                   for k in ("survival_rate", "escaped_A", "escaped_B", "dead", "steps_taken")}
                  if ppo_recs else {},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  저장: {csv_path}")
    print(f"  저장: {json_path}")


# ── 진입점 ────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험①: A* vs PPO 직접 비교")
    parser.add_argument("--episodes",   type=int, default=30)
    parser.add_argument("--scenarios",  type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--model-dir",  type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "model", "ppo"))
    parser.add_argument("--model-cls",  type=str, default="ppo",
                        choices=["ppo", "recurrent"], help="ppo | recurrent")
    parser.add_argument("--no-save",    action="store_true")
    args = parser.parse_args()

    all_astar, all_ppo = {}, {}
    for sc in args.scenarios:
        n = SCENARIO_CONFIGS[sc]["n_agents"]
        print(f"\n{'━'*66}")
        print(f"  시나리오 {sc}: {SCENARIO_CONFIGS[sc]['name']}  ({n}명)")
        print(f"{'━'*66}")

        print("\n[A* 베이스라인]")
        astar_recs = run_astar(sc, n, args.episodes)
        all_astar[sc] = astar_recs

        print(f"\n[{args.model_cls.upper()} 모델]")
        ppo_recs = run_ppo(sc, n, args.episodes, args.model_dir, args.model_cls)
        all_ppo[sc] = ppo_recs

        _print_comparison(sc, astar_recs, ppo_recs, args.model_cls)
        if not args.no_save:
            _save_results(sc, astar_recs, ppo_recs, args.model_cls)

    # 전체 요약
    print(f"\n{'═'*66}")
    print("  전체 시나리오 요약  (생존율 mean ± std)")
    print(f"{'─'*66}")
    print(f"  {'시나리오':<16} {'A* 베이스라인':>14}  {'PPO':>14}  {'차이':>8}")
    print(f"{'─'*66}")
    for sc in args.scenarios:
        a = np.array([r["survival_rate"] for r in all_astar[sc]])
        p_recs = all_ppo.get(sc, [])
        p = np.array([r["survival_rate"] for r in p_recs]) if p_recs else np.array([])
        name = SCENARIO_CONFIGS[sc]["name"]
        a_str = f"{a.mean():.1%}±{a.std():.1%}"
        p_str = f"{p.mean():.1%}±{p.std():.1%}" if len(p) > 0 else "N/A"
        diff  = f"+{p.mean()-a.mean():.1%}" if len(p) > 0 else "N/A"
        print(f"  S{sc} {name:<14} {a_str:>14}  {p_str:>14}  {diff:>8}")
    print(f"{'═'*66}")
