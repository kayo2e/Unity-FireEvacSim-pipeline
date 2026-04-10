# 🔥 Re-Filly: Stage 2 - Fire Evacuation Simulation

강화학습(PPO) 기반의 **지능형 유도등 제어**를 통한 화재 대피 최적화 시나리오 시뮬레이션입니다. 상상관 2층 구조를 그리드 맵으로 모델링하여 인원수 및 화재 상황별 대피 성능을 평가합니다.

---

## 📌 주요 특징
- **에이전트 속도 다양성**: 에이전트별로 랜덤한 속도($0.5 \sim 1.2$)를 부여하여 현실적인 군집 흐름 반영.
- **가변 인원 학습**: 10명, 30명, 50명 등 인구 밀도에 따른 최적 대피 경로 학습.
- **4단계 커리큘럼 시나리오**: 화재 확산, 출구 폐쇄 등 난이도별 단계적 학습 시스템.
- **PPO 알고리즘**: Stable Baselines3의 PPO를 사용하여 동적인 환경에서도 안정적인 정책 수립.
- **TensorBoard 모니터링**: 실시간 생존율 및 보상 그래프 시각화.

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

# 기본 학습 시작 (10, 30, 50명에 대해 각각 학습)
python train.py --mode train

# 특정 인원수와 스텝 설정
python train.py --mode train --people 30 --steps 1000000

# 30명 모델로 시나리오 3(출구 폐쇄) 테스트
python train.py --mode test --test-n 30 --test-scenario 3

tensorboard --logdir ./refilly_log/
```
