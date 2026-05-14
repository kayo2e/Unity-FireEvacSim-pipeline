"""
exp2_blocking.py — 실험 ②: 에피소드 중간 출구 차단 스트레스 테스트
====================================================================
에피소드 진행 중 특정 스텝에서 출구 하나를 강제 차단.
A*는 F1/F2 신호로 느리게 반응 → PPO는 학습된 패턴으로 빠르게 적응 예측.

핵심 비교 지표:
  - 차단 후 생존율   : PPO가 A*보다 높을수록 동적 적응력 증명
  - 탈출 출구 분배   : 차단 후 전환 속도 (escaped_A vs escaped_B)
  - 차단 없는 경우 대비 생존율 하락폭

실행:
    cd stage2
    python experiments/exp2_blocking.py
    python experiments/exp2_blocking.py --block-exit B --block-step 30
    python experiments/exp2_blocking.py --episodes 50 --scenarios 2 3 4
"""

import sys, os, json, csv, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baselines'))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_core import FireEvacEnv, SCENARIO_CONFIGS, EXIT_A_POS, EXIT_B_POS, WALL
from astar_real import astar_action

RESULT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "result", "exp2_blocking")


# ══════════════════════════════════════════════
# 중간 차단 환경 (env_core.py 수정 없이 서브클래스로 구현)
# ══════════════════════════════════════════════
class MidBlockFireEvacEnv(FireEvacEnv):
    """에피소드 도중 block_at_step에서 block_exit 출구를 강제 차단."""

    def __init__(self, *args, block_exit: str = "A",
                 block_at_step: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_exit    = block_exit.upper()
        self.block_at_step = block_at_step
        self._block_done   = False
        self._sim_step     = 0

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._block_done = False
        self._sim_step   = 0
        return result

    def step(self, action):
        # 차단 시점이 되면 그 스텝 실행 전에 적용
        if not self._block_done and self._sim_step >= self.block_at_step:
            self._apply_block()
            self._block_done = True

        obs, reward, terminated, truncated, info = super().step(action)
        self._sim_step += 1
        return obs, reward, terminated, truncated, info

    def _apply_block(self):
        target = EXIT_A_POS if self.block_exit == "A" else EXIT_B_POS
        for r, c in target:
            self.blocked_exits.add((r, c))
            self.grid[r, c] = WALL

        valid_A = [p for p in EXIT_A_POS if p not in self.blocked_exits]
        valid_B = [p for p in EXIT_B_POS if p not in self.blocked_exits]
        self._dist_to_exit_A = self._compute_bfs_specific(valid_A)
        self._dist_to_exit_B = self._compute_bfs_specific(valid_B)


# ══════════════════════════════════════════════
# A* 테스트
# ══════════════════════════════════════════════
def run_astar_blocking(scenario, n_agents, n_episodes, block_exit, block_at_step):
    cfg = SCENARIO_CONFIGS[scenario]
    records = []
    for ep in range(n_episodes):
        env = MidBlockFireEvacEnv(scenario=scenario, n_agents=n_agents,
                                   block_exit=block_exit, block_at_step=block_at_step)
        obs, _ = env.reset()
        total_r = 0.0
        info = {}
        for _ in range(cfg["max_steps"]):
            action = astar_action(env)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                break
        env.close()
        records.append(_make_rec(ep + 1, scenario, n_agents, info, total_r,
                                  block_exit, block_at_step))
        _print_ep(ep + 1, n_episodes, records[-1])
    return records


# ══════════════════════════════════════════════
# PPO 테스트
# ══════════════════════════════════════════════
def run_ppo_blocking(scenario, n_agents, n_episodes, model_dir, model_cls_name,
                     block_exit, block_at_step):
    cfg = SCENARIO_CONFIGS[scenario]
    ModelCls = _load_model_cls(model_cls_name)
    model_n, model_path, vecnorm_path = _find_model(model_dir, n_agents)
    if model_path is None:
        print(f"  [경고] {model_dir} 에 모델 없음 — PPO 테스트 건너뜀")
        return []

    print(f"  모델: {model_path}")
    model = ModelCls.load(model_path)
    records = []

    for ep in range(n_episodes):
        env = MidBlockFireEvacEnv(scenario=scenario, n_agents=n_agents,
                                   block_exit=block_exit, block_at_step=block_at_step)
        vec = DummyVecEnv([lambda _e=env: _e])
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

        records.append(_make_rec(ep + 1, scenario, n_agents, info, total_r,
                                  block_exit, block_at_step))
        _print_ep(ep + 1, n_episodes, records[-1])
    return records


# ══════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════
def _make_rec(ep, scenario, n_agents, info, total_r, block_exit, block_at_step):
    return {
        "episode":       ep,
        "scenario":      scenario,
        "scenario_name": SCENARIO_CONFIGS[scenario]["name"],
        "n_agents":      n_agents,
        "block_exit":    block_exit,
        "block_at_step": block_at_step,
        "survived":      info.get("escaped", 0),
        "escaped_A":     info.get("escaped_A", 0),
        "escaped_B":     info.get("escaped_B", 0),
        "dead":          info.get("dead", 0),
        "survival_rate": round(info.get("survival_rate", 0.0), 4),
        "steps_taken":   info.get("step", 0),
        "mean_panic":    round(info.get("mean_panic", 0.0), 4),
        "total_reward":  round(total_r, 2),
    }


def _print_ep(ep, total, rec):
    print(f"  [ep {ep:>3}/{total}] 생존율 {rec['survival_rate']:.0%} | "
          f"A:{rec['escaped_A']} B:{rec['escaped_B']} | {rec['steps_taken']}스텝")


def _stats(vals):
    a = np.array(vals, dtype=float)
    return {"mean": round(float(a.mean()), 4), "std":  round(float(a.std()),  4),
            "min":  round(float(a.min()),  4), "max":  round(float(a.max()),  4)}


def _load_model_cls(name):
    if name == "recurrent":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    from stable_baselines3 import PPO
    return PPO


def _find_model(model_dir, n_agents):
    if not os.path.isdir(model_dir):
        return None, None, None
    candidates = []
    for fname in os.listdir(model_dir):
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


def _print_comparison(sc, a_recs, p_recs, block_exit, block_at_step, ppo_label):
    cfg = SCENARIO_CONFIGS[sc]
    print(f"\n{'═'*70}")
    print(f"  S{sc} {cfg['name']}  |  출구{block_exit} 차단 @ 스텝{block_at_step}  "
          f"({cfg['n_agents']}명 × {len(a_recs)}회)")
    print(f"{'─'*70}")
    print(f"  {'지표':<18} {'A* 베이스라인':>14}  {'PPO (' + ppo_label + ')':>16}  {'차이':>8}")
    print(f"{'─'*70}")

    for label, key in [("생존율",   "survival_rate"),
                        ("탈출(A)", "escaped_A"),
                        ("탈출(B)", "escaped_B"),
                        ("사망자",  "dead"),
                        ("공황지수","mean_panic")]:
        a_vals = [r[key] for r in a_recs]
        p_vals = [r[key] for r in p_recs] if p_recs else []
        a_m = np.mean(a_vals)
        p_m = np.mean(p_vals) if p_vals else float("nan")
        diff = p_m - a_m if p_vals else float("nan")
        arrow = ("▲" if diff > 0 else "▼") if not np.isnan(diff) else " "
        if key == "survival_rate":
            print(f"  {label:<18} {a_m:>12.1%}  {p_m:>14.1%}  "
                  f"{arrow}{abs(diff):.1%}" if not np.isnan(diff) else
                  f"  {label:<18} {a_m:>12.1%}  {'N/A':>14}")
        else:
            print(f"  {label:<18} {a_m:>12.2f}  {p_m:>14.2f}  "
                  f"{arrow}{abs(diff):.2f}" if not np.isnan(diff) else
                  f"  {label:<18} {a_m:>12.2f}  {'N/A':>14}")
    print(f"{'═'*70}")


def _save(sc, a_recs, p_recs, ppo_label, block_exit, block_at_step):
    os.makedirs(RESULT_BASE, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"s{sc}_block{block_exit}{block_at_step}_{ts}"
    all_recs = [{"model": "astar", **r} for r in a_recs] + \
               [{"model": ppo_label, **r} for r in p_recs]
    if all_recs:
        with open(os.path.join(RESULT_BASE, f"exp2_{tag}.csv"),
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_recs[0].keys())
            writer.writeheader()
            writer.writerows(all_recs)
    summary = {
        "scenario": sc, "block_exit": block_exit, "block_at_step": block_at_step,
        "astar": {k: _stats([r[k] for r in a_recs])
                  for k in ("survival_rate", "escaped_A", "escaped_B", "dead")},
        "ppo":   {k: _stats([r[k] for r in p_recs])
                  for k in ("survival_rate", "escaped_A", "escaped_B", "dead")}
                 if p_recs else {},
    }
    with open(os.path.join(RESULT_BASE, f"exp2_{tag}_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {RESULT_BASE}/exp2_{tag}.*")


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험②: 중간 출구 차단 스트레스 테스트")
    parser.add_argument("--episodes",    type=int, default=30)
    parser.add_argument("--scenarios",   type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--block-exit",  type=str, default="A", choices=["A", "B"],
                        help="차단할 출구 (기본: A — 4셀 출구, 더 큰 충격)")
    parser.add_argument("--block-step",  type=int, default=50,
                        help="차단 발생 시뮬레이션 스텝 (기본: 50 = 에피소드 25% 지점)")
    parser.add_argument("--model-dir",   type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "model", "ppo"))
    parser.add_argument("--model-cls",   type=str, default="ppo",
                        choices=["ppo", "recurrent"])
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    print(f"\n실험②: 출구{args.block_exit} 강제 차단 (스텝 {args.block_step}) 스트레스 테스트")
    print(f"차단 시점: 에피소드 {args.block_step / 200 * 100:.0f}% 지점\n")

    for sc in args.scenarios:
        n = SCENARIO_CONFIGS[sc]["n_agents"]
        print(f"\n{'━'*70}")
        print(f"  시나리오 {sc}: {SCENARIO_CONFIGS[sc]['name']}  ({n}명)")

        print("\n[A* 베이스라인 — 차단 적용]")
        a_recs = run_astar_blocking(sc, n, args.episodes,
                                     args.block_exit, args.block_step)

        print(f"\n[{args.model_cls.upper()} — 차단 적용]")
        p_recs = run_ppo_blocking(sc, n, args.episodes, args.model_dir,
                                   args.model_cls, args.block_exit, args.block_step)

        _print_comparison(sc, a_recs, p_recs,
                          args.block_exit, args.block_step, args.model_cls)
        if not args.no_save:
            _save(sc, a_recs, p_recs, args.model_cls,
                  args.block_exit, args.block_step)
