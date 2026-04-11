# 🔥 Stage 2 - Intelligent Fire Evacuation System (Final)

본 프로젝트는 강화학습(**PPO**)을 활용하여 복잡한 건물 구조 내에서 최적의 대피 경로를 안내하는 **지능형 유도등 제어 시스템**입니다. 실제 상상관 2층 구조를 모델링하였으며, 화재 확산 및 출구 폐쇄 등 역동적인 재난 상황에서의 생존율 극대화를 목표로 합니다.

---

## ✨ Stage 2 & 3 주요 업데이트 (Implementation Complete)

### 1. 보상 체계 고도화 (Reward Shaping)
단순 탈출 여부뿐만 아니라 학습의 효율성을 위해 보상 함수를 정밀화했습니다.
- **Goal Reward**: EXIT 셀 실제 도달 시 **+20.0** (핵심 목표)
- **Distance Shaping**: 출구와의 거리가 단축되면 **+보상**, 멀어지면 **-패널티**를 부여하여 방황 방지
- **Survival Penalty**: 화재/연기 접촉 시 **-8.0**, 탈출 실패 시 잔류 인원당 **-3.0**

### 2. 맵 구조 및 연결성 개선 (Connectivity Fix)
- **양방향 경로 확보**: 기존에 엘리베이터(row 9~10)로 막혀있던 구간을 우회문 설치 및 6개 구역 추가 개방으로 수정.
- **완전 접근성**: 이제 모든 방에서 EXIT A와 EXIT B 양쪽으로 접근이 가능하며, AI는 화재 상황에 따라 유연하게 출구를 선택합니다.

### 3. 하이브리드 지능 및 성능 최적화
- **액션의 연속성**: RL이 `fire_cost_weight`(5.0~50.0) 파라미터를 직접 조절하여 실시간 위험 감수 수준을 결정합니다.
- **지능형 군중 제어**: 에이전트 밀도가 높은 셀에 추가 비용(`Density * 2.0`)을 할당하여 병목 현상을 방지합니다.
- **D* Lite 방식의 최적화**: 화재 맵의 변경이 감지될 때만 경로를 재계산하여 연산 효율을 극대화했습니다.

---

## 🧠 시스템 알고리즘 (Hybrid Intelligence)

AI와 전통적 알고리즘이 협력하는 2단계 구조로 안전성을 보장합니다.

1.  **High-Level (PPO)**: 건물 전체의 화재 확산과 인구 밀도를 관측하여 최적의 `FireCostWeight`를 결정합니다. (MlpPolicy 기반)
2.  **Low-Level (Dynamic BFS/Dijkstra)**: 결정된 가중치를 바탕으로 화재/연기/밀도를 고려한 최단·최안전 경로 방향을 에이전트에게 제시합니다.



---

## 🏗️ 커리큘럼 시나리오 (Curriculum Learning)

AI는 20회 에피소드 평균 생존율이 **50% 이상**일 때 다음 단계로 자동 승급합니다.

| 단계 | 시나리오 | 특징 |
| :--- | :--- | :--- |
| **Stage 1** | **초기 화재** | 고정 위치 소규모 화재 (기본 탈출 로직 습득) |
| **Stage 2** | **화재 확산** | 무작위 화재 발생 및 연기 확산 대응 |
| **Stage 3** | **출구 폐쇄** | 특정 출구 봉쇄 시 즉각적인 우회로 탐색 능력 |
| **Stage 4** | **폭발 및 붕괴** | 복도 폐쇄 및 극한 환경에서의 인원 분산 제어 |

---

## 🛠️ 설치 및 사용법 (Usage)

### 1. 필수 라이브러리 설치
```bash
pip install gymnasium stable-baselines3 tensorboard torch numpy

# [Check] 환경 로직 및 MlpPolicy 관측 규격(3600-dim) 검증
python fire_evac_final.py --mode check

# [Train] 10, 30, 50명 시나리오별 순차적 커리큘럼 학습 시작
python fire_evac_final.py --mode train --people 10 30 50 --steps 300000

# [Test] 학습된 모델로 테스트 (예: 50명 모델로 시나리오 4 테스트)
python fire_evac_final.py --mode test --test-n 50 --test-scenario 4

# [Monitor] TensorBoard 학습 지표 확인
tensorboard --logdir ./fire_evac_log/
```