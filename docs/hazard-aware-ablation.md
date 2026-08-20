# Hazard-Awareness Ablation — "화재 무시" 베이스라인이 실제로 화재를 무시하지 않았던 버그

## 발견 경위

정적 유도등 베이스라인을 추가하고 표 3에 반영하는 과정에서, "왜 정적 유도등의
생존율이 A*·PPO와 이렇게 근접한가?"라는 질문을 계기로 `env_core.py`를 다시
확인했다. `FireEvacEnv.step()`은 어떤 베이스라인이 만든 action이든 상관없이
항상 `_compute_dirs_for_strategy()` → `_compute_bfs_with_risk()`를 거치는데,
이 함수가 화재 셀 통과 비용(10.0)과 연기 비용(5.0)을 **무조건** 적용하고
있었다 — 베이스라인이 "화재를 무시한다"고 주장하든 말든 상관없이.

결과적으로 아래 세 베이스라인은 **이름·문서와 실제 동작이 어긋나 있었다**:

| 베이스라인 | 문서상 전제 | 실제 (수정 전) |
|---|---|---|
| `astar_simple_baseline.py` (Simple A*) | "화재 무시, 순수 최단거리" | 화재 셀 항상 회피 |
| `astar_real.py` (Pure A*) | "화재/연기/혼잡 무시" | 화재 셀 항상 회피 |
| `static_signage_baseline.py` (정적 유도등) | "화재·군중 상태와 무관하게 고정" | 화재 셀 항상 회피 |

`astar_baseline.py`(Hazard-aware A*)와 PPO는 애초에 화재 회피가 전제인
베이스라인이라 버그의 영향을 받지 않는다(의도한 대로 동작).

## 수정

`env_core.FireEvacEnv.__init__`에 `hazard_aware: bool = True` 파라미터를
추가했다. 기본값 True는 **기존 PPO 학습·평가와 Hazard-aware A*를 그대로
보존**한다(재학습 불필요, 하위 호환). `hazard_aware=False`로 생성하면
`_compute_bfs_with_risk`가 화재/연기 비용을 적용하지 않는다.

`astar_real.py`, `astar_simple_baseline.py`, `static_signage_baseline.py`,
그리고 `experiments/exp1_compare.py`의 A*/정적 유도등 실행 경로에
`hazard_aware=False`를 명시적으로 연결했다.

## Ablation 결과

### 사전 확인 (S4, n=10 에피소드, 수정 전후 비교)

| 지표 | 수정 전 (버그) | 수정 후 (실제 화재 무시) | 변화 |
|---|---:|---:|---:|
| A* 생존율 | 66.8% | 40.0% | **−26.8%p** |
| 정적 유도등 생존율 | 65.2% | 37.8% | **−27.4%p** |

버그가 있을 때는 A*·정적 유도등이 실제보다 훨씬 유리하게 보였다는 뜻이다.

### 전체 재측정 (S1~S5, n=30, seed=42 페어링) — 진행 중

<!-- 아래 표는 백그라운드 재실행 완료 후 채운다 -->
