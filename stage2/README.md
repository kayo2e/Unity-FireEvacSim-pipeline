# Stage 2 — 지능형 화재 대피 유도 시스템

강화학습(PPO)과 A* 규칙 기반 베이스라인을 비교하는 **지능형 유도등 제어 시스템**입니다.  
실제 상상관 2층 구조를 모델링하였으며, 화재 확산·군중 물리 시뮬레이션 환경에서 생존율 극대화를 목표로 합니다.

---

## 비교 구조

세 가지 방식으로 유도등 비용을 결정하며, 유도등→사람 이동 메커니즘(Dijkstra BFS)은 세 방식 모두 동일합니다.

| 방식 | 비용 결정 | 특징 |
| :--- | :--- | :--- |
| A* 규칙 기반 (베이스라인) | F1/F2 위협 스칼라 → 규칙 | 출구 혼잡도(F7/F8) 미인식, 분산유도 불가 |
| **RecurrentPPO (현재)** | LSTM 신경망 (lstm_hidden_size=256) | F7/F8/F10/F11로 혼잡·방향성 인식, 시계열 패턴 학습 |
| PPO + BC 사전학습 | A* 데모로 초기화 → PPO RL 파인튜닝 | A* 수준에서 출발 (BC가 A* 행동에 편향 유발 가능) |

> **정보 동등화**: A*와 PPO 모두 동일한 15개 스칼라 obs(F1~F15)만 입력으로 사용.  
> A*는 fire_map 직접 접근 없이 F1/F2 threshold 규칙만으로 동작.

```
환경 관측 (15차원 F1~F15)
        ↓
[비용 결정자: 규칙 / 신경망(RecurrentPPO+LSTM) / BC초기화 신경망]
        ↓
[exit_A_cost, exit_B_cost, crowd_weight]
        ↓
[Dijkstra BFS] — 화재·밀도 반영 경로 재계산
        ↓
[유도등] — 각 셀 방향 표시 (↑↓←→)
        ↓
[사람들이 유도등을 따라 이동]
```

---

## 관측 공간 (Observation Space)

**15차원 스칼라 피처 (F1~F15)** — 그리드 크기 독립, Unity 이식 가능

| 인덱스 | 피처 | 내용 | 값 범위 |
| :---: | :--- | :--- | :--- |
| 0 | F1 | 출구 A 화재 위협 (1=안전, 0=위험) — 화재→출구A BFS 최단거리 / 20 | 0.0 ~ 1.0 |
| 1 | F2 | 출구 B 화재 위협 (1=안전, 0=위험) — 화재→출구B BFS 최단거리 / 20 | 0.0 ~ 1.0 |
| 2 | F3 | 출구 A가 더 가까운 생존 인원 비율 | 0.0 ~ 1.0 |
| 3 | F4 | 탈출 완료 인원 비율 | 0.0 ~ 1.0 |
| 4 | F5 | 사망 인원 비율 | 0.0 ~ 1.0 |
| 5 | F6 | 시간 경과 비율 (긴급도) | 0.0 ~ 1.0 |
| 6 | **F7** | **출구 A 근접 혼잡도** — BFS 거리 4 이내 생존 인원 비율 ★ | 0.0 ~ 1.0 |
| 7 | **F8** | **출구 B 근접 혼잡도** — BFS 거리 4 이내 생존 인원 비율 ★ | 0.0 ~ 1.0 |
| 8 | F9 | 평균 공황 수준 (Helbing 2000) | 0.0 ~ 1.0 |
| 9 | **F10** | **생존자 전체의 출구 A까지 평균 BFS 거리 / 50** ★★ | 0.0 ~ 1.0 |
| 10 | **F11** | **생존자 전체의 출구 B까지 평균 BFS 거리 / 50** ★★ | 0.0 ~ 1.0 |
| 11 | F12 | 화재 무게중심 행 위치 / ROWS (공간 위치) | 0.0 ~ 1.0 |
| 12 | F13 | 화재 무게중심 열 위치 / COLS (공간 위치) | 0.0 ~ 1.0 |
| 13 | F14 | 출구 A 위협 변화율 — 이전 스텝 대비 F1 감소량 (화재 접근 속도) | 0.0 ~ 1.0 |
| 14 | F15 | 출구 B 위협 변화율 — 이전 스텝 대비 F2 감소량 (화재 접근 속도) | 0.0 ~ 1.0 |

> ★ **F7/F8 (병목 인식 피처)**: 출구 근방(반경 4) 혼잡을 실시간 관측.  
> A* 규칙 기반은 이 피처를 사용하지 않으므로 병목 상황에서 분산유도 불가.
>
> ★★ **F10/F11 (이동 방향성 피처)**: F7/F8이 "이미 출구에 몰린 인원 비율"이라면,  
> F10/F11은 "전체 생존자가 어느 출구 방향으로 이동 중인가"를 나타냅니다.  
> S3처럼 대규모 혼잡이 발생하기 전, PPO가 선제적으로 분산 유도를 결정할 수 있는 선행 신호입니다.

---

## 액션 공간 (Action Space)

**연속값 3차원**: `[exit_A_cost, exit_B_cost, crowd_weight]`

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
| 출구에서 1칸 멀어짐 | −2.0 | 매 스텝 (인원별) |
| 화재/연기 구역 진입 (사망) | **−20.0** | 매 스텝 (인원별) |
| 에피소드 종료 시 미탈출 인원 1명당 | **−15.0** | 에피소드 종료 1회 |
| 에피소드 종료 시 두 출구 모두 사용 | +15.0 | 에피소드 종료 1회 |

> **shaping reward 배율**: `urgency = 1.0 + (step / max_steps) × 2.0`  
> 시간이 지날수록 보상 강도가 ×1.0 → ×3.0으로 증가하여 타임아웃을 억제합니다.
>
> **출구 분산 보너스**: 두 출구를 균형 있게 활용하도록 학습을 유도합니다.  
> A* 규칙 기반은 학습이 없으므로 이 보너스를 직접적으로 활용하지 않습니다.

---

## 병목(Bottleneck) 시뮬레이션

실제 피난 환경의 출구·복도 처리량 한계를 반영하며, 분산유도 학습에 실질적인 의미를 부여합니다.

| 파라미터 | 값 | 근거 |
| :--- | :---: | :--- |
| `EXIT_CAPACITY` | **1명/셀/스텝** | 출구당 스텝당 탈출 인원 엄격 제한 → 명수별 병목 차이 강화 |
| `CELL_CAPACITY` | **1명/셀** | Fruin(1971): 복도 밀도 초과 시 이동 차단 → 실제 혼잡 발생 |
| `QUEUE_RADIUS` | BFS 거리 4 이내 | 혼잡도 피처(F7/F8) 측정 기준 |

```
[이동 시도] → EXIT 셀인가?
                ├─ YES: 이번 스텝 exit_quota 남음? → YES: 탈출 성공
                │                                 → NO:  대기 (제자리)
                └─ NO:  목적지 셀 점유 < CELL_CAPACITY? → YES: 이동
                                                        → NO:  대기 (혼잡 차단)
```

### PPO vs A* — 분산유도 능력 비교

| 항목 | PPO | A* (베이스라인) |
| :--- | :--- | :--- |
| 출구 혼잡도 인식 | F7/F8 피처로 실시간 관측 | **인식 불가** |
| 출구별 비용 차별화 | `exit_A_cost` / `exit_B_cost` 독립 조정 | 화재 거리만으로 결정 |
| 병목 발생 시 대응 | 혼잡한 출구 비용↑ → 반대편으로 분산 | 모든 인원이 가까운 출구에 집중 |
| 분산 보상 활용 | 두 출구 모두 사용 시 +15 수신 | 동일 보상 구조 (학습 없음) |

---

## 커리큘럼 학습 (Curriculum Learning)

최근 **50 에피소드** 평균 생존율 **≥ 85%** 달성 시 다음 단계 자동 승급.

| 단계 | 시나리오 | 화재 위치 | 확산 | 스텝 | 인원 | PPO vs A* 예상 |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | 기본 탈출 | (2,1) 고정 | 5% | 200 | 10 | 동률 (sanity check) |
| 2 | 점진적 위협 | 중앙 복도 무작위 | 25% | 200 | 15 | PPO 소폭 우세 (F14/F15 선제 대응) |
| 3 | 일방 과부하 | Exit B 구역(80%) + 중앙(80%) 독립 점화 | 22% | 250 | **30** | **PPO 우세** (F7/F8 혼잡 분산) |
| 4 | 부분 위협 | 출구 A·B 구역 각 60% 독립 점화 | 18% | 280 | **15** | **PPO 우세** (위협+혼잡 동시 trade-off) |

> **S2~S4 인원수**: 권장 인원수가 기본값. `--n` 미지정 시 자동 적용됨.  
> **S3 점화**: Exit B 위협으로 전원이 Exit A로 쏠리고, 중앙 화재가 북상하는 인원을 추격.  
> **S4 점화 분포**: 양쪽 동시 위협 36% / 단일 위협 48% / 무위협 16%로 복합 상황 혼재.

---

## 모델 구조

### 알고리즘 비교

| 항목 | **PPO** (`ppo_train.py`) | **RecurrentPPO** (`recurrent_ppo_train.py`) |
| :--- | :--- | :--- |
| 라이브러리 | stable-baselines3 | sb3_contrib |
| Policy | MlpPolicy | MlpLstmPolicy |
| 네트워크 | MLP `[256, 256]` | LSTM(256) + MLP `[256, 256]` |
| 모델 크기 | ~1.7 MB | ~9.5 MB |
| 입력 차원 | 15 (F1~F15) | 15 + LSTM hidden state |
| 출력 분포 | Gaussian 3차원 | Gaussian 3차원 |

### 공통 하이퍼파라미터

| 파라미터 | 값 | 설명 |
| :--- | :---: | :--- |
| `n_steps` | 2048 | 환경당 롤아웃 스텝 |
| `batch_size` | 256 | 미니배치 크기 |
| `n_epochs` | 10 | 롤아웃당 업데이트 반복 횟수 |
| `gamma` | 0.99 | 할인율 |
| `learning_rate` | 3e-4 | Adam 학습률 |
| `clip_range` | 0.2 | PPO 클리핑 범위 |
| `ent_coef` | 0.05 | 엔트로피 보너스 |
| `max_grad_norm` | 0.5 | 그래디언트 클리핑 |
| 총 학습 스텝 | **3,000,000** | BC 없음 (--bc-steps 0) |

### 학습 환경

| 항목 | 값 |
| :--- | :--- |
| 병렬 환경 | SubprocVecEnv (Windows: DummyVecEnv) |
| 관측 정규화 | VecNormalize (norm_obs=True, clip_obs=10.0) |
| 커리큘럼 기준 | 최근 50 에피소드 평균 생존율 ≥ 85% → 다음 단계 승급 |
| 커리큘럼 인원 | S1=10명 → S2=15명 → S3=30명 → S4=15명 (자동 전환) |

```
입력 (15차원 F1~F15)
    ↓
[PPO]  Linear(15→256) → ReLU → Linear(256→256) → ReLU
[RPPO] LSTM(256) → Linear(256→256) → ReLU → Linear(256→256) → ReLU
    ↓
┌──────────────────┬──────────────────┐
Actor head         Critic head
Linear(256→3)      Linear(256→1)
Gaussian 샘플링     가치 추정 V(s)
    ↓
[exit_A_cost, exit_B_cost, crowd_weight]
```

---

## BC 사전학습 (A* → PPO 초기화)

학습 시작 전 A* 규칙 행동을 시연 데이터로 수집해 PPO 신경망을 지도학습으로 초기화합니다.  
이후 PPO RL로 파인튜닝하여 A* 수준에서 출발해 분산유도까지 추가 학습합니다.

```
collect_astar_demos()  → 12,000 샘플 수집 (기본 3000스텝 × 4환경)
pretrain_bc()          → MSE 지도학습으로 PPO 신경망 초기화 (기본 5에폭)
model.learn()          → PPO RL 파인튜닝
```

- `DummyVecEnv` + `VecNormalize`로 데모 수집 → BC 입력이 학습 분포와 일치
- `dist.distribution.loc` (Gaussian mean)을 타깃으로 MSE 최소화
- 그래디언트 클리핑 0.5 — PPO `max_grad_norm`과 동일
- `--bc-steps 0`으로 BC 비활성화 가능 (순수 PPO 단독 학습)

---

## 군중 물리 모델

| 모델 | 적용 내용 |
| :--- | :--- |
| Fruin (1971) | 밀도 기반 이동 속도 감소 — `CELL_CAPACITY` 초과 시 이동 차단 |
| Helbing (2000) | 공황(panic) 레벨 업데이트 — F9 피처로 관측, 사망 시 주변 공황 전파 |

---

## 파일 구조

```
stage2/
├── train.py                            # 메인 학습/테스트 스크립트 (BC 사전학습 포함)
├── astar_baseline.py                   # A* 규칙 기반 베이스라인 (비교용)
├── env_core.py                         # 환경 로직 (FireEvacEnv, 상수, 시나리오)
├── fire_evac_model_{n}ppl.zip          # 학습된 PPO 모델 (n=인원수)
├── fire_evac_model_{n}ppl_vecnorm.pkl  # VecNormalize 정규화 통계
├── fire_evac_log/                      # TensorBoard 로그
│   └── PPO_{n}ppl_{k}/
└── result/
    ├── results/                        # PPO 테스트 결과
    └── astar_baseline/                 # A* 베이스라인 테스트 결과
```

---

## 설치 및 실행

```bash
pip install gymnasium stable-baselines3 sb3-contrib tensorboard torch numpy
```

```bash
# 환경 검증
python train.py --mode check

# 학습 — BC 비활성화 (RecurrentPPO 단독, 현재 권장)
python train.py --mode train --people 10 --steps 300000 --bc-steps 0

# BC 사전학습 포함 (기본값: bc-steps=3000, bc-epochs=5)
python train.py --mode train --people 10 --steps 1000000

# BC 설정 조정
python train.py --mode train --people 10 --steps 1000000 --bc-steps 5000 --bc-epochs 10

# 여러 인원수 순차 학습
python train.py --mode train --people 10 30 50 --steps 1000000

# 학습된 모델 테스트
python train.py --mode test --test-n 10 --test-scenario 1 --test-episodes 30

# 그리드 렌더링 켜기
python train.py --mode test --test-n 10 --test-scenario 2 --render

# 저장 없이 터미널 출력만
python train.py --mode test --test-n 10 --no-save

# TensorBoard 학습 지표 확인
tensorboard --logdir ./fire_evac_log/
```

```bash
# A* 규칙 기반 베이스라인 테스트 (비교용)
python astar_baseline.py --scenario 1 --n 10 --episodes 30
python astar_baseline.py --all-scenarios --n 10 --episodes 30
```

---

## 테스트 결과 저장

`--mode test` 실행 시 `results/` 폴더에 자동 저장됩니다.

```
results/test_results_s{scenario}_{n}ppl_{timestamp}.csv   # 에피소드별 원본 기록
results/test_summary_s{scenario}_{n}ppl_{timestamp}.json  # 통계 요약
```

**CSV 컬럼**

| 컬럼 | 설명 |
| :--- | :--- |
| `survival_rate` | 에피소드별 생존율 |
| `total_reward` | 누적 보상 |
| `steps_taken` | 종료까지 스텝 수 |
| `escaped` / `dead` / `remaining` | 탈출/사망/잔류 인원 |
| `escaped_A` / `escaped_B` | 출구 A·B별 탈출 인원 (분산유도 정량 지표) |
| `mean_panic` | 평균 공황 수준 |
| `max_fire_cells` | 최대 화재 셀 수 |

**통계 항목** (mean / std / min / max / median): `survival_rate`, `total_reward`, `steps_taken`, `escaped`, `escaped_A`, `escaped_B`, `dead`, `mean_panic`

---

## 성능 비교

> 공통 조건: 각 시나리오 **30 에피소드**, `EXIT_CAPACITY=1` / `CELL_CAPACITY=1` 병목 적용  
> 측정일: 2026-05-11 | 학습: **3,000,000 스텝** | 커리큘럼 threshold=0.85 / window=50 | BC 없음

### 알고리즘 비교 (생존율 mean ± std)

| 시나리오 | 인원 | A* 규칙 기반 | **PPO** (MlpPolicy) | **RecurrentPPO** (MlpLstmPolicy) |
| :---: | :---: | :---: | :---: | :---: |
| S1 기본 탈출 | 10명 | 100.0% ± 0.0% | **100.0% ± 0.0%** | **100.0% ± 0.0%** |
| S2 점진적 위협 | 15명 | 93.3% ± 7.2% | **93.3% ± 7.8%** | 92.2% ± 9.1% |
| S3 일방 과부하 | 30명 | **71.6% ± 22.9%** | 68.4% ± 17.0% | 70.4% ± 17.1% |
| S4 부분 위협 | 15명 | 62.0% ± 40.7% | **72.0% ± 30.1%** | 68.0% ± 33.5% |

> **해석**:  
> - S1·S2: 세 알고리즘 모두 동등 (차이 < std 범위)  
> - S3: A*가 오히려 소폭 우세 — RL이 복잡한 혼잡 시나리오에서 A*를 아직 뛰어넘지 못함  
> - **S4가 핵심**: PPO +10%p (72.0% vs 62.0%) — 양쪽 출구 동시 위협 상황에서 RL의 분산유도 우위 발현  
> - **PPO vs RecurrentPPO**: 모든 시나리오에서 차이가 std 이내 → LSTM이 성능 개선 원인이 아님  
> - **결론: 성능 개선의 주 원인은 학습 스텝 수(3M)**

```bash
# A* 베이스라인
python astar_baseline.py --all-scenarios --episodes 30

# PPO 테스트
python ppo_train.py --mode test --model-n 10 --all-scenarios --test-episodes 30

# RecurrentPPO 테스트
python recurrent_ppo_train.py --mode test --model-n 10 --all-scenarios --test-episodes 30
```

---

## 개선 이력

### 관측 공간 변경

| 버전 | 차원 | 비고 |
| :--- | :--- | :--- |
| 초기 | 3600 | 6채널 × 40×25 flatten |
| 중간 | 3606 | + 요약 피처 F1~F6 |
| 중간 | 3608 | + F7/F8 병목 혼잡도 |
| 중간 | 9 | F1~F9 스칼라만 — 그리드 독립, Unity 이식 가능 |
| 중간 | 15 | + F10~F15 (혼잡 절댓값, 화재 위치, 위협 변화율) |
| **현재** | **15** | F10/F11 재정의 — 생존자 전체의 출구별 평균 BFS 거리 |

### 학습 안정성 (NaN 크래시 수정)

| 항목 | 변경 전 | 변경 후 |
| :--- | :--- | :--- |
| `learning_rate` | 5e-4 | 3e-4 |
| `max_grad_norm` | 미설정 | 0.5 명시 |
| `VecNormalize` | 없음 | norm_obs=True, clip=10 |
| BFS 정규화 | `/ mx` | `/ max(mx, 1.0)` |

### 주요 설계 변경

| 항목 | 변경 전 | 변경 후 |
| :--- | :--- | :--- |
| 액션 공간 | `fire_cost` 1차원 | `[exit_A_cost, exit_B_cost, crowd_weight]` 3차원 |
| 출구 분산 보상 | 없음 | 두 출구 모두 사용 시 +15 |
| 병목 시뮬레이션 | 없음 (무제한) | EXIT_CAPACITY=1 / CELL_CAPACITY=1 |
| BC 사전학습 | 없음 | A* 데모 → PPO 신경망 초기화 후 RL 파인튜닝 |
| 군중 물리 | 없음 | Fruin(1971) 속도 감소 + Helbing(2000) 공황 모델 |
| shaping reward | 고정 ×2.0 | urgency ×1.0 → ×3.0 (시간 경과 비례) |
| 사망 패널티 | −8.0 / 미탈출 −5.0 | **−20.0 / 미탈출 −15.0** (생존 우선 강화) |
| 커리큘럼 기준 | window=20 / threshold=50% | **window=50 / threshold=85%** (충분한 수렴 후 진급) |
| 커리큘럼 인원 버그 | 전 시나리오 동일 n_agents | **시나리오별 권장 인원수 자동 적용** |
| 모델 | PPO (MlpPolicy) | **RecurrentPPO (MlpLstmPolicy, lstm_hidden_size=256)** |
| n_steps | 1024 | **2048** (긴 에피소드 GAE 추정 안정화) |
| F10/F11 | 출구 근방 절댓값 혼잡도 | **생존자 전체의 출구별 평균 BFS 거리** (이동 방향성) |
| 사망 판정 | 이동 전후 이중 적용 | **화재 확산 후 단일 판정** (이중 계산 제거) |
| Potential shaping | 없음 | 화재-출구 거리 변화 ×1.5 (접근 시 패널티) |
