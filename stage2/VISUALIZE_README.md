# visualize_episode.py — Episode Grid Visualization

에피소드를 실행하고 스텝별 그리드 이미지를 PNG/GIF로 저장합니다.

---

## 빠른 시작

```bash
cd stage2

# BFS 베이스라인, 시나리오 1, 10스텝마다 저장
python3 visualize_episode.py --scenario 1 --baseline bfs --every 10

# Simple A*, 시나리오 4, 전체 스텝 저장
python3 visualize_episode.py --scenario 4 --baseline simple_astar --every 1

# 학습된 모델 (자동 탐색)
python3 visualize_episode.py --scenario 2 --baseline model

# 전체 시나리오 한번에
python3 visualize_episode.py --all-scenarios --baseline bfs --every 5
```

---

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--scenario` | `1` | 시나리오 번호 (1~4) |
| `--all-scenarios` | — | 시나리오 1~4 전부 실행 |
| `--n` | 시나리오 권장값 | 인원수 |
| `--baseline` | `bfs` | 정책 선택 (아래 표 참고) |
| `--model-path` | 자동 탐색 | 모델 `.zip` 경로 (`--baseline model` 시) |
| `--every` | `1` | N스텝마다 프레임 저장 (1=전체) |
| `--no-gif` | — | GIF 생성 안 함 |
| `--episode` | `1` | 출력 폴더명에 붙는 에피소드 번호 |

### `--baseline` 선택지

| 값 | 파일 | 설명 |
|----|------|------|
| `bfs` | `baselines/astar_baseline.py` | 화재 셀 차단 BFS |
| `astar` | `baselines/astar_real.py` | 순수 A* (Manhattan, 화재 무시) |
| `simple_astar` | `baselines/astar_simple_baseline.py` | 순수 A* (화재 무시, 단순 구현) |
| `model` | `model/recurrent_ppo/` or `model/ppo/` | 학습된 PPO/RecurrentPPO 모델 |

모델 자동 탐색 순서: `model/recurrent_ppo/s{N}_best.zip` → `model/ppo/s{N}_best.zip`

---

## 출력 파일

저장 위치: `result/visualize/s{시나리오}_{정책}_ep{에피소드}/`

| 파일 | 설명 |
|------|------|
| `frame_XXXX.png` | 스텝별 개별 그리드 이미지 |
| `episode.gif` | 전체 에피소드 애니메이션 |
| `summary.png` | 초기 / 중간 / 최종 3장 비교 이미지 |

---

## 그리드 범례

| 색상 | 의미 |
|------|------|
| 검정 | 벽 (Wall) |
| 연회색 + 화살표 | 복도 + 유도등 방향 |
| 베이지 | 방 (Room) |
| 초록 | 출구 A (Exit A) |
| 짙은 초록 | 출구 B (Exit B) |
| 짙은 빨강 | 차단된 출구 (Blocked Exit) |
| 빨강/주황 | 화재 (Fire) |
| 회색 | 연기 (Smoke) |
| 파란 점 | 사람 (Person) |

---

## 시나리오별 권장 설정

| 시나리오 | 권장 인원 | 특징 |
|----------|-----------|------|
| S1 | 20명 | 기본, 화재 느림 |
| S2 | 20명 | 출구 하나 차단 |
| S3 | 30명 | 복잡한 화재 확산 |
| S4 | 40명 | 대규모, 빠른 화재 |

`--every 5~10` 권장 (S4는 스텝 수 많아 `--every 1` 시 100개 이상 프레임 생성)