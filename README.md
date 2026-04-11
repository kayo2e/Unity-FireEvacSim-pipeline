# 🔥  Stage 2 - Fire Evacuation Simulation

강화학습(PPO) 기반의 **지능형 유도등 제어**를 통한 화재 대피 최적화 시나리오 시뮬레이션입니다. 상상관 2층 구조를 그리드 맵으로 모델링하여 인원수 및 화재 상황별 대피 성능을 평가합니다.

---

## 📌 주요 특징
- **에이전트 속도 다양성**: 에이전트별로 랜덤한 속도($0.5 \sim 1.2$)를 부여하여 현실적인 군집 흐름 반영.
- **가변 인원 학습**: 10명, 30명, 50명 등 인구 밀도에 따른 최적 대피 경로 학습.
- **4단계 커리큘럼 시나리오**: 화재 확산, 출구 폐쇄 등 난이도별 단계적 학습 시스템.
- **PPO 알고리즘**: Stable Baselines3의 PPO를 사용하여 동적인 환경에서도 안정적인 정책 수립.
- **TensorBoard 모니터링**: 실시간 생존율 및 보상 그래프 시각화.

## 🧠 지능형 전략 제어 (Strategic Control)
기존의 개별 유도등 제어 방식에서 벗어나, AI가 실시간 재난 상황을 분석하여 최적의 대피 전략을 선택하는 **하이브리드 제어 방식**을 채택했습니다.

- **Action Space 최적화**: 300여 개의 개별 유도등 방향 결정($4^{300}$) → 3가지 핵심 대피 전략 선택($3^1$)으로 복잡도를 획기적으로 낮춤.
- **전략 구성**:
  1. **Strategy 0 (EXIT A Priority)**: 상단 출구(EXIT A) 중심의 최단 경로 유도.
  2. **Strategy 1 (EXIT B Priority)**: 하단 출구(EXIT B) 중심의 최단 경로 유도.
  3. **Strategy 2 (All-Exit Balanced)**: 화재 상황에 따라 모든 가용 출구로 인원 분산 유도.
- **알고리즘 결합**: AI(PPO)가 고수준 전략을 결정하면, 저수준에서는 BFS(Breadth-First Search) 알고리즘이 미리 계산된 정밀한 방향 맵을 즉시 적용하여 안전성을 보장함.

---

## 🏗️ 시뮬레이션 환경 (Environment)

### 1. 맵 정보 (Grid Map)
- **Size**: $30 \times 20$ (상상관 2층 실측 기반)
- **요소**: 벽(`█`), 출구(`E`), 화재(`F`), 연기(`~`), 에이전트(`P`), 유도등(`↑↓→←`)

### 2. 커리큘럼 시나리오 구성
| 단계 | 시나리오 명 | 주요 제약 사항 |
| :--- | :--- | :--- |
| **Stage 1** | 초기 화재 | 특정 위치 소규모 화재 발생 |
| **Stage 2** | 화재 확산 | 화재 및 연기의 무작위 확산 |
| **Stage 3** | 출구 폐쇄 | 주요 대피로 및 출구 이용 불가 상황 발생 |
| **Stage 4** | 폭발 및 붕괴 | 무작위 경로 폐쇄 및 극한의 탈출 난이도 |

---

## 🛠️ 설치 및 사용법 (Usage)

### 1. 필수 라이브러리 설치
```bash
pip install gymnasium stable-baselines3 tensorboard torch numpy

# 환경 검증 (관측치 및 유도등 설정 확인)
python train.py --mode check

# 전체 인원수(10, 30, 50명) 순차 학습
python train.py --mode train

# 특정 설정으로 학습 (예: 30명, 100만 스텝)
python train.py --mode train --people 30 --steps 1000000

# 학습된 모델 테스트 (예: 30명 모델로 시나리오 3 테스트)
python train.py --mode test --test-n 30 --test-scenario 3

tensorboard --logdir ./fire_evac_log/
```

