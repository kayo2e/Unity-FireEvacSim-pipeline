"""
exp_speed_breakdown.py — 전체 제어주기 시간 분해 측정
================================================================================
교수 검토 메모(한국시뮬레이션학회논문지_확인 필요 내용.docx, 3.1절 "우선순위 1") 대응.

기존 exp_speed.py의 비교는 측정 범위가 서로 다르다 — PPO 쪽은 model.predict()
(신경망 forward pass)만 재고, Hazard-aware BFS 쪽은 bfs_action()(생존자별 BFS
재탐색 전체)을 잰다. "정책 추론 대 전체 경로탐색"을 비교한 것이라 "전체 제어
속도"의 공정한 비교가 아니다(같은 지적: needcheck 메모 2.1절, 3.3절).

이 스크립트는 제안 정책의 전체 제어주기를 4단계로 분해해 각각 재므로, PPO의
"전체 제어주기" 시간도 알 수 있고, Hazard-aware BFS의 "경로탐색 전체" 시간과
어느 구간을 비교하는 게 공정한지 판단할 근거를 제공한다.

  T_control = T_observation + T_policy + T_costmap + T_guidance

  T_observation : env._get_obs() — 4차원 관측 벡터 구성
  T_policy      : model.predict() — PPO 신경망 forward pass
  T_costmap     : env._compute_dirs_for_strategy() 내부의 두 Dijkstra 비용장 계산
                  (exit_a_cost/exit_b_cost/crowd_weight로부터 D_A, D_B 생성)
  T_guidance    : 위 비용장으로부터 각 보행 가능 셀의 유도 방향을 결정하는 부분
                  (현재 구현에서는 _compute_dirs_for_strategy() 안에 costmap과
                  함께 섞여 있어 실측으로는 분리되지 않는다 — 아래 CAVEAT 참고)

CAVEAT(측정 한계, 반드시 본문에 명시할 것):
  - _compute_dirs_for_strategy()가 Dijkstra 비용장 생성과 방향 결정을 한
    함수 안에서 같이 수행해, 현재 코드 구조로는 T_costmap과 T_guidance를
    별도 타이머로 분리할 수 없다. 이 스크립트는 둘을 합쳐 T_costmap_guidance로
    보고한다. 완전히 분리하려면 _compute_dirs_for_strategy()를 두 함수로
    리팩터링해야 하며, 이번 측정에서는 수행하지 않았다.
  - Hazard-aware BFS(bfs_action())는 자체적으로 "관측 구성"과 "경로탐색"이
    분리되지 않은 단일 함수라, 같은 4단계로 분해할 수 없다. BFS는 전체
    bfs_action() 1회 호출 시간만 측정해 T_control(PPO)의 합계와 비교한다
    (부분 구간끼리의 비교가 아니라 "전체 대 전체" 비교임을 명시).

실행:
    cd stage2
    python experiments/exp_speed_breakdown.py --scenarios 4 --steps 300 \
        --feature-set reduced --n-agents-list 20 50 100 150 200 300 500
"""
import sys, os, time, argparse, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baselines'))

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from astar_baseline import bfs_action


def _find_model(model_dir, n_agents, feature_set="full"):
    from stable_baselines3 import PPO as _PPO
    expected_dim = 15 if feature_set == "full" else 4
    candidates = []
    for f in (os.listdir(model_dir) if os.path.isdir(model_dir) else []):
        if f.endswith(".zip") and "ppl" in f and "best" not in f:
            try:
                n = int(f.replace("fire_evac_model_", "").replace("ppl.zip", ""))
            except ValueError:
                continue
            path = os.path.join(model_dir, f"fire_evac_model_{n}ppl.zip")
            try:
                if _PPO.load(path).observation_space.shape == (expected_dim,):
                    candidates.append(n)
            except Exception:
                pass
    if not candidates:
        return None, None
    best = min(candidates, key=lambda x: abs(x - n_agents))
    path = os.path.join(model_dir, f"fire_evac_model_{best}ppl.zip")
    return path, path.replace(".zip", "_vecnorm.pkl")


def bench_ppo_breakdown(model_dir, scenario, n_agents, n_steps, feature_set="full"):
    from stable_baselines3 import PPO

    mpath, vpath = _find_model(model_dir, n_agents, feature_set)
    if mpath is None:
        return None

    model = PPO.load(mpath)
    env = FireEvacEnv(scenario=scenario, n_agents=n_agents, feature_set=feature_set)
    vec = DummyVecEnv([lambda: env])
    if vpath and os.path.exists(vpath):
        vec = VecNormalize.load(vpath, vec)
        vec.training = False
        vec.norm_reward = False

    obs = vec.reset()
    t_obs, t_policy, t_costmap_guidance = [], [], []

    for _ in range(n_steps):
        # T_observation: env._get_obs()를 직접 재호출해 측정(reset/step이
        # 반환한 obs는 이미 계산 완료된 결과라 별도 재호출로 시간만 측정하고
        # F14/F15 델타 상태 오염을 피하려고 env.reset()으로 깨끗한 복사본을
        # 쓰지 않는 대신, 실제 로직과 동일한 _get_obs() 호출 자체의 비용만 잰다.
        t0 = time.perf_counter()
        _ = env._get_obs()
        t_obs.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        t_policy.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        env._compute_dirs_for_strategy(
            float(action[0][0]), float(action[0][1]), float(action[0][2]))
        t_costmap_guidance.append(time.perf_counter() - t0)

        obs, _, done, _ = vec.step(action)
        if done[0]:
            obs = vec.reset()

    vec.close()
    return {
        "observation_ms": np.array(t_obs) * 1000,
        "policy_ms": np.array(t_policy) * 1000,
        "costmap_guidance_ms": np.array(t_costmap_guidance) * 1000,
    }


def bench_hazard_bfs_total(scenario, n_agents, n_steps):
    env = FireEvacEnv(scenario=scenario, n_agents=n_agents, hazard_aware=True)
    env.reset()
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        bfs_action(env)
        times.append(time.perf_counter() - t0)
        action = bfs_action(env)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset()
    return np.array(times) * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, nargs="+", default=[4])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--feature-set", choices=["full", "reduced"], default="reduced")
    parser.add_argument("--n-agents-list", type=int, nargs="+",
                         default=[20, 50, 100, 150, 200, 300, 500])
    parser.add_argument("--model-dir", type=str,
                         default=os.path.join(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__))), "model", "ppo"))
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    rows = []
    for sc in args.scenarios:
        for n in args.n_agents_list:
            print(f"\n[S{sc} {SCENARIO_CONFIGS[sc]['name']} — {n}명]")
            bd = bench_ppo_breakdown(args.model_dir, sc, n, args.steps, args.feature_set)
            if bd is None:
                print("  [경고] 모델 없음, 건너뜀")
                continue
            t_obs = bd["observation_ms"].mean()
            t_pol = bd["policy_ms"].mean()
            t_cmg = bd["costmap_guidance_ms"].mean()
            t_control = t_obs + t_pol + t_cmg
            t_bfs = bench_hazard_bfs_total(sc, n, args.steps).mean()

            print(f"  T_observation        : {t_obs:.4f} ms")
            print(f"  T_policy             : {t_pol:.4f} ms")
            print(f"  T_costmap+guidance   : {t_cmg:.4f} ms")
            print(f"  T_control (합계)      : {t_control:.4f} ms")
            print(f"  Hazard-aware BFS 전체 : {t_bfs:.4f} ms  (참고: PPO의 T_policy만이 "
                  f"아니라 T_control과 비교해야 공정함)")

            rows.append({
                "scenario": sc, "n_agents": n,
                "observation_ms": t_obs, "policy_ms": t_pol,
                "costmap_guidance_ms": t_cmg, "control_total_ms": t_control,
                "hazard_bfs_total_ms": t_bfs,
            })

    out_csv = args.out_csv or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "result", "exp_speed_breakdown.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"\n저장: {out_csv}")
