# Stage 2 — 강화학습 기반 유도등 제어 상세

최상위 [README](../README.md)에서 전체 파이프라인 및 결과를 확인하세요.

---

## 환경 파라미터

| 항목 | 값 |
| :--- | :--- |
| 맵 크기 | 40 × 25 셀 (실제 상상관 2층 기반) |
| 출구 A | 행 10~11, 열 18~21 (4셀) |
| 출구 B | 행 34~35, 열 10~11 (2셀) |
| 병목 제한 | `EXIT_CAPACITY=1` / `CELL_CAPACITY=1` |
| 군중 물리 | Fruin (1971) 밀도 속도 감소 + Helbing (2000) 공황 모델 |

---

## 액션 공간

연속값 3차원: `[exit_A_cost, exit_B_cost, crowd_weight]`

| 차원 | 범위 | 설명 |
| :--- | :--- | :--- |
| `exit_A_cost` | 5.0 ~ 50.0 | 출구 A 경로 화재 회피 강도 |
| `exit_B_cost` | 5.0 ~ 50.0 | 출구 B 경로 화재 회피 강도 |
| `crowd_weight` | 0.5 ~ 5.0 | 밀도 패널티 강도 (클수록 분산 유도) |

---

## 모델 하이퍼파라미터

| 파라미터 | 값 |
| :--- | :---: |
| 정책 | MlpPolicy (PPO, stable-baselines3) |
| 네트워크 | MLP [256, 256] |
| `n_steps` | 2048 |
| `batch_size` | 256 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `learning_rate` | 3e-4 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.05 |
| 총 학습 스텝 | **3,000,000** |

---

## 실행 명령어

```bash
# PPO 커리큘럼 학습
python train_ppo_grid.py --mode train --steps 3000000

# 시나리오별 A* vs PPO 비교 실험
python experiments/exp1_compare.py

# 추론 속도 비교
python experiments/exp_speed.py

# 에피소드 GIF 시각화
python experiments/exp3_visualize.py

# A* 베이스라인
python baselines/astar_baseline.py --all-scenarios --episodes 30
python baselines/astar_simple_baseline.py --all-scenarios --episodes 30
python baselines/astar_real.py --all-scenarios --episodes 30

# 포스터용 그래프 생성
python plot_exp1_table.py          # 표 3
python plot_speed_cached.py        # 그림 5
python poster_grid.py              # 그림 4 경로 비교 그리드

# TensorBoard
tensorboard --logdir ./fire_evac_log/
```
