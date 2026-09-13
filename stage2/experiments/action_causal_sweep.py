"""
action_causal_sweep.py — 액션(exit_A_cost, exit_B_cost, crowd_weight)이 생존율에
실제로 인과적 영향을 주는지, 학습된 정책과 무관하게 직접 확인한다.
================================================================================
지금까지의 발견(F1~F15 마스킹 무의미, log_std 발산, 정책 액션이 항상
최솟값 근처)은 "정책이 안 배웠다"는 가설을 뒷받침하지만, 다른 가설도 있다:
"애초에 이 액션이 생존율에 거의 영향을 못 준다(환경의 hazard_aware=True
기본값이 이미 화재 회피를 다 해줘서)". 후자가 맞다면 정책을 아무리
재학습해도 지금 이상으로는 안 나온다 — 재학습 성공 여부를 가르는 핵심 진단.

고정된(학습되지 않은) 액션 값 몇 개를 매 스텝 그대로 사용해 롤아웃하고,
hazard_aware=True/False 두 조건에서 생존율이 액션에 따라 얼마나 달라지는지
직접 측정한다. 액션 자체의 인과 효과 크기를 정책 학습 여부와 분리해서 본다.

실행:
    cd stage2
    python experiments/action_causal_sweep.py --scenario 2 --episodes 15
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_core import FireEvacEnv, SCENARIO_CONFIGS

ACTION_CONFIGS = {
    "corner_min(5,5,0.5)":      (5.0, 5.0, 0.5),    # 지금 학습된 정책이 항상 내는 값
    "prefer_A(5,50,0.5)":       (5.0, 50.0, 0.5),   # A를 극단적으로 선호
    "prefer_B(50,5,0.5)":       (50.0, 5.0, 0.5),   # B를 극단적으로 선호
    "high_crowd(5,5,5.0)":      (5.0, 5.0, 5.0),    # 혼잡 회피 최대
    "mid(27,27,2.5)":           (27.5, 27.5, 2.5),  # 중간값
}


def run_fixed_action(scenario, n_agents, n_episodes, base_seed, action, hazard_aware):
    cfg = SCENARIO_CONFIGS[scenario]
    action_arr = np.array(action, dtype=np.float32)
    rates = []
    for ep in range(n_episodes):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents, hazard_aware=hazard_aware)
        seed = base_seed + ep
        env.reset(seed=seed)
        info = {}
        for _ in range(cfg["max_steps"]):
            _, _, term, trunc, info = env.step(action_arr)
            if term or trunc:
                break
        env.close()
        rates.append(info.get("survival_rate", 0.0))
    return np.array(rates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_agents = SCENARIO_CONFIGS[args.scenario]["n_agents"]
    print(f"=== S{args.scenario} ({SCENARIO_CONFIGS[args.scenario]['name']}), "
          f"n={n_agents}, episodes={args.episodes} ===\n")

    results = {}
    for hazard in [True, False]:
        print(f"\n--- hazard_aware={hazard} ---")
        results[hazard] = {}
        for name, action in ACTION_CONFIGS.items():
            rates = run_fixed_action(args.scenario, n_agents, args.episodes,
                                      args.seed, action, hazard)
            results[hazard][name] = (rates.mean(), rates.std())
            print(f"  {name:<22} 생존율 {rates.mean():.1%} ± {rates.std():.1%}")

    print(f"\n{'='*70}\n요약: 액션에 따른 생존율 차이 (같은 hazard_aware 내에서 최대-최소)\n{'='*70}")
    for hazard in [True, False]:
        vals = [v[0] for v in results[hazard].values()]
        spread = max(vals) - min(vals)
        print(f"  hazard_aware={hazard}: 최대 {max(vals):.1%}, 최소 {min(vals):.1%}, "
              f"스프레드 {spread:.1%}p")

    print(f"\n{'='*70}\n판정\n{'='*70}")
    spread_true = max(v[0] for v in results[True].values()) - min(v[0] for v in results[True].values())
    if spread_true < 0.05:
        print("  hazard_aware=True 환경에서 액션을 바꿔도 생존율 스프레드가 5%p 미만.")
        print("  -> 지금 액션 설계로는 정책이 아무리 학습해도 큰 이득을 내기 어려움.")
        print("  -> hazard_aware를 끄거나(정책이 직접 위험을 반영하게) 액션의 레버리지를")
        print("     키우는 구조 변경이 필요.")
    else:
        print(f"  hazard_aware=True에서도 액션에 따라 {spread_true:.1%}p 차이가 남.")
        print("  -> 액션이 실제로 인과적 영향을 준다는 뜻 — 지금 도는 재학습이 성공할")
        print("     여지가 있음. 정책이 이 차이를 학습해서 활용하는지가 관건.")
