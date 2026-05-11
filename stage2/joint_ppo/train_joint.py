"""
train_joint.py — JointPPO (경로 Transformer + MaskablePPO) 학습/테스트
=======================================================================
기존 ppo_train.py / recurrent_ppo_train.py 와 완전 분리.
모델: model/joint_ppo/
결과: result/joint_ppo/
로그: logs/joint_ppo/

설치: pip install sb3-contrib

실행:
    python joint_ppo/train_joint.py --mode train --people 10 --steps 300000
    python joint_ppo/train_joint.py --mode test --model-n 10 --all-scenarios --test-episodes 30
"""

import sys
import os

_STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STAGE2)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# CPU에서 소형 행렬(seq=65) 연산 시 멀티스레드 오버헤드를 방지.
# GPU 사용 시 이 설정은 무시된다.
import torch
torch.set_num_threads(1)

try:
    from sb3_contrib import MaskablePPO
except ImportError:
    raise SystemExit("sb3_contrib 미설치. pip install sb3-contrib")

import numpy as np
import csv
import json
import gymnasium as gym
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from env_core import SCENARIO_CONFIGS
from joint_ppo.env_joint import JointEvacEnv, K_MAX
from joint_ppo.policy_joint import PathTransformerPolicy

BASE_DIR   = _STAGE2
MODEL_DIR  = os.path.join(BASE_DIR, "model",  "joint_ppo")
RESULT_DIR = os.path.join(BASE_DIR, "result", "joint_ppo")
LOG_DIR    = os.path.join(BASE_DIR, "logs",   "joint_ppo")


# ══════════════════════════════════════════════
# 커리큘럼 래퍼
# ══════════════════════════════════════════════
class JointCurriculumWrapper(gym.Wrapper):
    """생존율 임계치 도달 시 시나리오를 순서대로 진급."""

    def __init__(self, n_agents: int = None, threshold: float = 0.85,
                 window: int = 50, k_max: int = K_MAX):
        self.current_scenario = 1
        self._k_max           = k_max
        s1_n = SCENARIO_CONFIGS[1]["n_agents"] if n_agents is None else n_agents
        self.n_agents         = s1_n
        env = JointEvacEnv(scenario=1, n_agents=s1_n, k_max=k_max)
        super().__init__(env)
        self.threshold = threshold
        self.window    = window
        self.recent: list = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if term or trunc:
            self.recent.append(info["survival_rate"])
            if len(self.recent) > self.window:
                self.recent.pop(0)
            avg = sum(self.recent) / len(self.recent)
            if (len(self.recent) == self.window
                    and avg >= self.threshold
                    and self.current_scenario < len(SCENARIO_CONFIGS)):
                self.current_scenario += 1
                cfg_n = SCENARIO_CONFIGS[self.current_scenario]["n_agents"]
                self.env = JointEvacEnv(
                    scenario=self.current_scenario, n_agents=cfg_n,
                    k_max=self._k_max)
                self.n_agents = cfg_n
                self.recent   = []
                print(f"\n[커리큘럼] ★ {self.current_scenario}단계 승급! "
                      f"({self.env.cfg['name']}) | 생존율 {avg:.0%}")
        return obs, rew, term, trunc, info

    def reset(self, **kw):
        return self.env.reset(**kw)

    def action_masks(self):
        return self.env.action_masks()

    @property
    def cfg(self):
        return self.env.cfg


# ══════════════════════════════════════════════
# 학습 콜백
# ══════════════════════════════════════════════
class JointCallback(BaseCallback):
    def __init__(self, log_interval: int = 10_000):
        super().__init__()
        self.log_interval  = log_interval
        self.ep_survival: list = []
        self.ep_rewards:  list = []
        self._cum_rewards = None
        self._ep_count    = 0
        self._last_summary_step = 0

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]
        if self._cum_rewards is None:
            self._cum_rewards = np.zeros(len(rewards))
        self._cum_rewards += rewards

        new_ep = False
        for i, done in enumerate(dones):
            if done:
                self.ep_rewards.append(float(self._cum_rewards[i]))
                self._cum_rewards[i] = 0.0
                new_ep = True
                self._ep_count += 1

        for info in self.locals.get("infos", []):
            if "survival_rate" in info:
                self.ep_survival.append(info["survival_rate"])

        # 에피소드 종료마다 현재 줄을 실시간 덮어쓰기
        if new_ep and self.ep_survival:
            n_ep  = min(len(self.ep_survival), 20)
            avg_s = sum(self.ep_survival[-n_ep:]) / n_ep
            last  = self.ep_survival[-1]
            sc    = getattr(getattr(self.training_env, "envs", [None])[0],
                            "current_scenario", "?")
            print(f"\r  [ep {self._ep_count:>5}] S{sc} | "
                  f"이번 {last:>5.1%} | "
                  f"이동평균(20) {avg_s:>5.1%} | "
                  f"Step {self.num_timesteps:>8,}",
                  end="", flush=True)

        # 10,000스텝마다 고정 요약 줄 출력
        if (self.num_timesteps - self._last_summary_step >= self.log_interval
                and self.ep_survival):
            self._last_summary_step = self.num_timesteps
            n     = min(len(self.ep_survival), 100)
            avg_s = sum(self.ep_survival[-n:]) / n
            avg_r = (sum(self.ep_rewards[-min(len(self.ep_rewards), n):]) / n
                     if self.ep_rewards else 0.0)
            print(f"\n  ── Step {self.num_timesteps:>8,} | "
                  f"생존율(100ep) {avg_s:>5.1%} | 보상 {avg_r:>+7.1f}")
        return True


# ══════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════
def train(person_counts=(10,), total_timesteps: int = 300_000,
          n_envs: int = 4, d_model: int = 64,
          nhead: int = 4, num_layers: int = 2,
          k_max: int = K_MAX):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 62)
    print("화재대피유도시스템 — JointPPO (경로 Transformer) 학습")
    print(f"Policy    : PathTransformerPolicy")
    print(f"           k_max={k_max} | d_model={d_model} | nhead={nhead} | layers={num_layers}")
    print(f"           seq={k_max+1} 토큰 → CPU 가능")
    print(f"디바이스  : {device} | 병렬 환경: {n_envs}개")
    print(f"모델 저장 : {MODEL_DIR}")
    print("=" * 62)

    for n in person_counts:
        s1_n = SCENARIO_CONFIGS[1]["n_agents"]
        print(f"\n{'─'*62}\n커리큘럼 학습 시작 "
              f"(S1 {s1_n}명 → 이후 시나리오별 인원수 자동 적용)\n{'─'*62}")

        def _make_env(seed: int):
            def _init():
                env = JointCurriculumWrapper(k_max=k_max)
                env.reset(seed=seed)
                return env
            return _init

        raw_vec = DummyVecEnv([_make_env(i) for i in range(n_envs)])
        vec_env = VecNormalize(raw_vec, norm_obs=True, norm_reward=False,
                               clip_obs=10.0)

        model = MaskablePPO(
            PathTransformerPolicy, vec_env,
            device          = device,
            verbose         = 0,
            n_steps         = 2048,
            batch_size      = 256,
            n_epochs        = 10,
            gamma           = 0.99,
            learning_rate   = 3e-4,
            clip_range      = 0.2,
            ent_coef        = 0.05,
            max_grad_norm   = 0.5,
            policy_kwargs   = dict(
                k_max      = k_max,
                d_model    = d_model,
                nhead      = nhead,
                num_layers = num_layers,
            ),
            tensorboard_log = LOG_DIR,
        )

        model.learn(
            total_timesteps = total_timesteps,
            callback        = JointCallback(),
            tb_log_name     = f"JointPPO_{n}ppl",
        )

        save_path = os.path.join(MODEL_DIR, f"fire_evac_model_{n}ppl")
        model.save(save_path)
        vec_env.save(save_path + "_vecnorm.pkl")
        vec_env.close()
        print(f"\n모델 저장: {save_path}.zip")

    print(f"\n학습 완료! TensorBoard: tensorboard --logdir {LOG_DIR}")


# ══════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════
def _stats(vals: list) -> dict:
    a = np.array(vals, dtype=float)
    return {
        "mean":   round(float(a.mean()),        4),
        "std":    round(float(a.std()),          4),
        "min":    round(float(a.min()),          4),
        "max":    round(float(a.max()),          4),
        "median": round(float(np.median(a)),     4),
    }


def test(n_agents: int = 10, scenario: int = 1,
         n_episodes: int = 30, model_n: int = None,
         save_results: bool = True, render: bool = False):

    model_n    = model_n if model_n is not None else n_agents
    model_path = os.path.join(MODEL_DIR, f"fire_evac_model_{model_n}ppl")
    vecnorm_path = model_path + "_vecnorm.pkl"

    print(f"\n모델 로드: {model_path}.zip  (테스트 인원: {n_agents}명)")
    model = MaskablePPO.load(model_path)

    render_mode = "human" if render else None
    env     = JointEvacEnv(scenario=scenario, n_agents=n_agents,
                            render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: env])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"정규화 통계 로드: {vecnorm_path}")

    records = []
    for ep in range(n_episodes):
        obs      = vec_env.reset()
        total_r  = 0.0
        step_cnt = 0
        max_fire = 0
        print(f"\n[에피소드 {ep+1}/{n_episodes}] "
              f"{env.cfg['name']} | {n_agents}명", end="", flush=True)

        for _ in range(env.cfg["max_steps"]):
            # action_masks를 vec_env를 통해 수집
            from sb3_contrib.common.maskable.utils import get_action_masks
            masks  = get_action_masks(vec_env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, r, done, infos = vec_env.step(action)
            info      = infos[0]
            total_r  += float(r[0])
            step_cnt  = info["step"]
            max_fire  = max(max_fire, info["fire_cells"])
            if render:
                env.render()
            if done[0]:
                break

        rec = {
            "episode":       ep + 1,
            "scenario":      scenario,
            "scenario_name": env.cfg["name"],
            "n_agents":      n_agents,
            "escaped":       info["escaped"],
            "escaped_A":     info.get("escaped_A", 0),
            "escaped_B":     info.get("escaped_B", 0),
            "dead":          info["dead"],
            "remaining":     info["remaining"],
            "survival_rate": round(info["survival_rate"], 4),
            "total_reward":  round(total_r, 2),
            "steps_taken":   step_cnt,
            "max_fire_cells":max_fire,
            "mean_panic":    round(info.get("mean_panic", 0.0), 4),
            "blocked_exits": str(info["blocked_exits"]),
        }
        records.append(rec)
        print(f" → 탈출 {rec['escaped']}/{n_agents} | "
              f"생존율 {rec['survival_rate']:.0%} | "
              f"보상 {rec['total_reward']:+.1f} | {step_cnt}스텝")

    vec_env.close()

    summary = {
        "model":         model_path,
        "scenario":      scenario,
        "scenario_name": env.cfg["name"],
        "n_agents":      n_agents,
        "n_episodes":    n_episodes,
        "survival_rate": _stats([r["survival_rate"]  for r in records]),
        "total_reward":  _stats([r["total_reward"]   for r in records]),
        "steps_taken":   _stats([r["steps_taken"]    for r in records]),
        "escaped":       _stats([r["escaped"]        for r in records]),
        "escaped_A":     _stats([r["escaped_A"]      for r in records]),
        "escaped_B":     _stats([r["escaped_B"]      for r in records]),
        "dead":          _stats([r["dead"]           for r in records]),
        "mean_panic":    _stats([r["mean_panic"]     for r in records]),
        "max_fire_cells":_stats([r["max_fire_cells"] for r in records]),
    }

    print("\n" + "═" * 62)
    print(f"  테스트 결과 | {env.cfg['name']} | {n_agents}명 × {n_episodes}회")
    print("═" * 62)
    for key in ("survival_rate", "total_reward", "steps_taken",
                "escaped", "dead", "mean_panic"):
        s = summary[key]
        print(f"  {key:<16} mean={s['mean']:>8}  std={s['std']:>7}  "
              f"min={s['min']:>7}  max={s['max']:>7}")
    print("═" * 62)

    if save_results:
        os.makedirs(RESULT_DIR, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag       = f"s{scenario}_{n_agents}ppl_{ts}"
        csv_path  = os.path.join(RESULT_DIR, f"test_results_{tag}.csv")
        json_path = os.path.join(RESULT_DIR, f"test_summary_{tag}.json")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n  에피소드 기록: {csv_path}")
        print(f"  통계 요약   : {json_path}")

    return records, summary


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JointPPO 화재대피 학습/테스트")
    parser.add_argument("--mode",          choices=["train", "test"], default="train")
    parser.add_argument("--people",        type=int, nargs="+", default=[10])
    parser.add_argument("--steps",         type=int, default=300_000)
    parser.add_argument("--n-envs",        type=int, default=4)
    parser.add_argument("--test-n",        type=int, default=10)
    parser.add_argument("--test-scenario", type=int, default=1)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--model-n",       type=int, default=None)
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--d-model",       type=int, default=64)
    parser.add_argument("--nhead",         type=int, default=4)
    parser.add_argument("--num-layers",    type=int, default=2)
    parser.add_argument("--k-max",         type=int, default=K_MAX,
                        help=f"제어할 최대 경로 셀 수 (기본 {K_MAX})")
    args = parser.parse_args()

    if args.mode == "train":
        train(
            person_counts  = args.people,
            total_timesteps= args.steps,
            n_envs         = args.n_envs,
            d_model        = args.d_model,
            nhead          = args.nhead,
            num_layers     = args.num_layers,
            k_max          = args.k_max,
        )

    elif args.mode == "test":
        scenarios    = [1, 2, 3, 4] if args.all_scenarios else [args.test_scenario]
        all_summaries = {}
        for sc in scenarios:
            n = SCENARIO_CONFIGS[sc]["n_agents"] if args.all_scenarios else args.test_n
            _, summary = test(
                n_agents     = n,
                scenario     = sc,
                n_episodes   = args.test_episodes,
                model_n      = args.model_n,
                save_results = not args.no_save,
                render       = args.render,
            )
            all_summaries[sc] = summary

        if args.all_scenarios:
            print(f"\n{'═'*62}")
            print("  전체 시나리오 생존율 (JointPPO)")
            print(f"{'─'*62}")
            for sc, s in all_summaries.items():
                sr = s["survival_rate"]
                print(f"  S{sc} {s['scenario_name']:<8} | "
                      f"생존율 {sr['mean']:.1%} ± {sr['std']:.1%}  "
                      f"(min {sr['min']:.0%} ~ max {sr['max']:.0%})")
            print(f"{'═'*62}")
