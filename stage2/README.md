# Stage 2 - 지능형 화재 대피 유도 시스템

강화학습(PPO)을 활용하여 복잡한 건물 구조 내에서 최적의 대피 경로를 안내하는 **지능형 유도등 제어 시스템**입니다.  
실제 상상관 2층 구조를 모델링하였으며, 화재 확산 등 역동적인 재난 상황에서의 생존율 극대화를 목표로 합니다.

---

## 시스템 구조

AI와 전통적 경로 탐색 알고리즘이 협력하는 2단계 하이브리드 구조입니다.

```
[PPO 에이전트] ──→ fire_cost_weight 결정 (5.0 ~ 50.0)
                         ↓
[Dijkstra BFS]  ──→ 화재/연기/밀도를 반영한 최적 경로 계산
                         ↓
[유도등 307개]  ──→ 각 셀의 방향 표시 (↑↓←→)
                         ↓
              [사람들이 유도등을 따라 이동]
```

- **High-Level (PPO)**: 건물 전체의 화재·연기·인구밀도를 관측(3606-dim flatten)하여 위험 회피 가중치 결정
- **Low-Level (Dijkstra)**: 결정된 가중치로 화재 셀 패널티를 조정, 실시간 경로 재계산

---

## 관측 공간 (Observation Space)

6채널 × 30행 × 20열 + 요약 피처 6개 = **3606차원** (MlpPolicy용 flatten)

**공간 채널 (3600차원)**

| 채널 | 내용 | 값 범위 |
| :--- | :--- | :--- |
| ch 0 | 건물 구조 (벽/복도/출구 등) | 0.0 ~ 1.0 |
| ch 1 | 화재 맵 | 0 or 1 |
| ch 2 | 연기 맵 | 0 or 1 |
| ch 3 | 사람 밀도 (정규화) | 0.0 ~ 1.0 |
| ch 4 | 유도등 방향 | 0.0 ~ 1.0 |
| ch 5 | BFS 출구 거리 (정규화, 가까울수록 1.0) | 0.0 ~ 1.0 |

**요약 피처 (6차원)** — MLP가 3600픽셀에서 직접 추출하기 어려운 핵심 정보를 사전 계산해 제공

| 인덱스 | 내용 | 값 범위 |
| :--- | :--- | :--- |
| 3600 | 출구 A 화재 위협 (0=위험, 1=안전) | 0.0 ~ 1.0 |
| 3601 | 출구 B 화재 위협 (0=위험, 1=안전) | 0.0 ~ 1.0 |
| 3602 | 출구 A 쪽이 더 가까운 생존 인원 비율 | 0.0 ~ 1.0 |
| 3603 | 탈출 완료 인원 비율 | 0.0 ~ 1.0 |
| 3604 | 사망 인원 비율 | 0.0 ~ 1.0 |
| 3605 | 시간 경과 비율 (긴급도) | 0.0 ~ 1.0 |

---

## 액션 공간 (Action Space)

- **연속값 3차원**: `[exit_A_cost, exit_B_cost, crowd_weight]`

| 차원 | 범위 | 설명 |
| :--- | :--- | :--- |
| `exit_A_cost` | 5.0 ~ 50.0 | 출구 A 경로의 화재 회피 강도 (클수록 우회) |
| `exit_B_cost` | 5.0 ~ 50.0 | 출구 B 경로의 화재 회피 강도 (클수록 우회) |
| `crowd_weight` | 0.5 ~ 5.0 | 밀도 패널티 강도 (클수록 분산 유도) |

---

## 보상 함수 (Reward)

| 이벤트 | 보상 | 적용 시점 |
| :--- | :--- | :--- |
| EXIT 셀 실제 도달 | +20.0 | 매 스텝 (인원별) |
| 출구 방향으로 1칸 접근 | +2.0 | 매 스텝 (인원별) |
| 출구에서 1칸 멀어짐 | -2.0 | 매 스텝 (인원별) |
| 화재/연기 구역 진입 (사망) | -8.0 | 매 스텝 (인원별) |
| 에피소드 종료 시 미탈출 인원 1명당 | -5.0 | 에피소드 종료 1회 |
| 에피소드 종료 시 두 출구 모두 사용 | +15.0 | 에피소드 종료 1회 |

> **미탈출 패널티**: 탈출 못한 인원 전체(`사망자 + 잔류자`)에 적용됩니다.  
> 사망으로 `terminated`되거나 타임아웃으로 `truncated`되는 경우 모두 적용되어  
> 생존율을 직접적으로 높이는 방향으로 학습을 유도합니다.
>
> **출구 분산 보너스**: A* 규칙 기반 대비 차별화 포인트. A*는 개인 최단경로를 따르므로 인원이 한 출구로 쏠리는 경향이 있습니다. PPO가 두 출구를 균형있게 활용하도록 학습을 유도합니다.
>
> **shaping reward 배율**: `urgency = 1.0 + (step / max_steps) × 2.0` 으로 시간이 지날수록 보상 강도가 ×1.0 → ×3.0으로 증가하여 타임아웃을 억제합니다.

---

## 커리큘럼 학습 (Curriculum Learning)

20 에피소드 평균 생존율 **≥ 50%** 달성 시 다음 단계 자동 승급.  
16개 병렬 환경이 각자 독립적으로 승급하기 때문에 출력에 같은 승급 메시지가 여러 번 표시됩니다.

| 단계 | 시나리오 | 화재 위치 | 확산 확률 | 최대 스텝 | 특징 |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | 초기 화재 | (3,1) 고정 | 5% | 200 | 기본 탈출 로직 습득 |
| 2 | 화재 확산 | 무작위 2~3곳 | 12% | 250 | 무작위 화재·연기 대응 |
| 3 | 출구 위협 | (8,10) 고정 — Exit A 바로 옆 | 25% | 350 | 출구가 불에 막힐 때 대체 출구 탐색 |
| 4 | 폭발 붕괴 | 무작위 3~6곳 | 20% | 600 | 복도 붕괴 30% + 출구 폐쇄 30% 확률 |

> **시나리오 3 설계 의도**: 불이 Exit A 바로 옆에서 시작해 빠르게 확산됩니다.  
> 에이전트는 "언제 Exit A를 포기하고 Exit B로 전원 유도할지"를 학습합니다.  
> **시나리오 4 max_steps**: 복도 붕괴로 우회 경로가 길어지므로 400 → 600으로 상향했습니다.

---

## 개선 내역

### 학습 안정성 (NaN 크래시 수정)

기존 학습에서 **NaN 크래시**(78% 지점)가 발생하여 다음을 수정했습니다.

| 항목 | 변경 전 | 변경 후 | 이유 |
| :--- | :--- | :--- | :--- |
| `learning_rate` | 5e-4 | 3e-4 | gradient explosion 방지 |
| `max_grad_norm` | 미설정 | 0.5 명시 | gradient clipping 보장 |
| `VecNormalize` | 없음 | norm_obs=True, clip=10 | NN 입력 안정화 |
| BFS 정규화 | `/ mx` | `/ max(mx, 1.0)` | divide-by-zero 방지 |

### 생존율 개선 (시나리오 3·4 대응)

| 항목 | 변경 전 | 변경 후 | 이유 |
| :--- | :--- | :--- | :--- |
| shaping reward 기준 거리 | reset 시 1회 계산 (정적) | 화재 변경 시마다 재계산 (동적) | 화재로 막힌 출구 방향에 `+` 보상이 주어지는 신호 충돌 제거 |
| shaping 거리 계산 방식 | 화재 무시 BFS | 화재 셀을 벽으로 처리한 BFS | 유도등 방향과 shaping reward 방향 일치 |
| 시나리오 4 `max_steps` | 400 | 600 | 복도 붕괴 시 우회 경로가 길어져 시간 부족 해소 |
| 액션 공간 | `fire_cost` 1차원 | `[exit_A_cost, exit_B_cost, crowd_weight]` 3차원 | 출구별 독립 제어로 선제 분산 유도 학습 가능 |
| shaping reward 배율 | 고정 `× 2.0` | `× 2.0 × urgency` (step 0 → ×1.0, 마지막 → ×3.0) | 타임아웃 방지: 시간 경과할수록 출구 방향 신호 강화 |
| 관측 공간 | 3600차원 (6채널 flatten) | 3606차원 (+ 요약 피처 6개) | 화재 위협·인원 분포 등 핵심 정보 사전 계산 제공 |
| 출구 분산 보상 | 없음 | 에피소드 종료 시 두 출구 모두 사용 시 +15 | A* 대비 차별화: 글로벌 인원 분배 학습 유도 |

---

## 파일 구조

```
stage2/
├── train.py                            # 메인 학습/테스트 스크립트
├── astar_baseline.py                   # A* 규칙 기반 베이스라인 (비교용)
├── fire_evac_model_{n}ppl.zip          # 학습된 PPO 모델 (n=인원수)
├── fire_evac_model_{n}ppl_vecnorm.pkl  # VecNormalize 정규화 통계
├── fire_evac_log/                      # TensorBoard 로그
│   └── PPO_{n}ppl_{k}/
└── result/
    ├── test_ppo_{n}ppl{k}/             # PPO 테스트 결과
    └── astar_baseline/                 # A* 베이스라인 테스트 결과
```

---

## 설치 및 실행

```bash
pip install gymnasium stable-baselines3 tensorboard torch numpy
```

```bash
# 환경 검증
python train.py --mode check

# 학습 (10명, 100만 스텝 권장)
python train.py --mode train --people 10 --steps 1000000

# 여러 인원수 순차 학습
python train.py --mode train --people 10 30 50 --steps 1000000

# 학습된 모델 테스트 (기본 10에피소드, 결과 저장)
python train.py --mode test --test-n 10 --test-scenario 1

# 에피소드 수 지정
python train.py --mode test --test-n 10 --test-scenario 1 --test-episodes 30

# 그리드 렌더링 켜기
python train.py --mode test --test-n 10 --test-scenario 2 --render

# 저장 없이 터미널 출력만
python train.py --mode test --test-n 10 --no-save

# 구 모델(3600차원 obs) 호환 테스트 — 신규 학습 전 기존 모델 평가 시 사용
python train.py --mode test --test-n 10 --test-scenario 3 --test-episodes 30 --legacy-obs

# TensorBoard 학습 지표 확인
tensorboard --logdir ./fire_evac_log/
```

> **`--legacy-obs` 플래그**: obs 차원이 3600 → 3606으로 변경되어 기존 학습된 모델은 새 obs와 호환되지 않습니다.  
> 기존 모델을 테스트할 때는 `--legacy-obs`를 붙여 3600차원 모드로 실행하세요.  
> 신규 학습(`--mode train`)에는 사용하지 않습니다.

```bash
# A* 규칙 기반 베이스라인 테스트 (비교용)
python astar_baseline.py --all-scenarios --n 10 --episodes 30
```

---

## 테스트 결과 저장

`--mode test` 실행 시 자동으로 아래 파일이 생성됩니다.

```
test_results_s{scenario}_{n}ppl_{timestamp}.csv   # 에피소드별 원본 기록
test_summary_s{scenario}_{n}ppl_{timestamp}.json  # 통계 요약
```

**CSV 컬럼**

| 컬럼 | 설명 |
| :--- | :--- |
| `survival_rate` | 에피소드별 생존율 |
| `total_reward` | 누적 보상 |
| `steps_taken` | 종료까지 스텝 수 |
| `escaped` / `dead` / `remaining` | 탈출/사망/잔류 인원 |
| `max_fire_cells` | 최대 화재 셀 수 |
| `blocked_exits` | 막힌 출구 목록 |

**통계 항목** (mean / std / min / max / median): `survival_rate`, `total_reward`, `steps_taken`, `escaped`, `dead`

---

## 학습 결과 (10명, 300,000 스텝)

```
Step   10,000 | 생존율 55.6% | 보상  +294.1   (1~2단계 전환 구간)
Step   20,000 | 생존율 38.4% | 보상  +...     (3~4단계 진입, 어려워짐)
Step  300,000 | 생존율 32.0% | 보상  +...     (4단계 폭발·붕괴 학습 중)
```

**커리큘럼 진행**: 1단계(100%) → 2단계(86~98%) → 3단계(72~82%) → 4단계(20~37%)

> 4단계(폭발 붕괴)는 300k 스텝으로 수렴이 부족합니다.  
> 충분한 학습을 위해 `--steps 1000000` 이상을 권장합니다.