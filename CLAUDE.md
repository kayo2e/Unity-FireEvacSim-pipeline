# Unity-FireEvacSim-pipeline

피난안내도 그리드 자동 추출(Stage 1) + 강화학습(PPO) 기반 유도등 제어(Stage 2)를 통한
화재 대피 시뮬레이션. A* 베이스라인 3종과 비교, KCI 저널 게재를 목표로 한다.

# Workflow

## Git

### Branch 전략

- `main`에 직접 커밋하지 않는다 — 항상 작업용 브랜치를 새로 만들어서 거기서 커밋한다.
- 브랜치 이름은 커밋 카테고리와 맞춰 접두어를 붙인다: `feat/`, `fix/`, `docs/`, `refactor/`,
  `chore/`, `perf/` (예: `fix/exp1-table-labels`).
- 작업이 끝나면 push만 하고, `main`으로의 머지는 사용자 확인 후 진행한다 — 자동 머지하지 않는다.

### Commit Template

`<category>: <short_summary>`

- categories: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`
- 요약은 한국어로 쓴다 (이 저장소의 기존 커밋 관례를 따름) — 영어 강제하지 않음
- 본문(body)은 꼭 필요할 때만: 여러 항목을 바꾼 큰 커밋이면 `- ` bullet로 정리
- **Claude를 co-author로 넣지 않는다** — `Co-Authored-By: Claude` 트레일러, "Generated with
  Claude Code" 같은 문구를 커밋 메시지나 PR 본문에 절대 추가하지 않는다 (GitHub Contributors에
  Claude가 표시되지 않도록 하려는 명시적 요청, 2026-08-20)

## 데이터 무결성

- README에 적힌 실험 수치(표 3 등)는 반드시 `stage2/result/` 아래의 raw CSV/JSON과 대조해서
  일치하는지 확인한 뒤에만 신뢰한다 — 과거에 라벨링 실수(시나리오 번호 밀림)와 근거 없는
  수치가 섞여 들어간 적이 있었다.
- 시나리오 비교 실험(`experiments/exp1_compare.py`)은 `--seed`로 A*/PPO를 같은 화재·시작
  조건에 페어링해서 돌린다 — 재현성 확보 + paired 유의성 검정(`scipy.stats.ttest_rel`,
  `wilcoxon`)이 가능해짐. 시드 없는 결과와 섞어서 비교하지 않는다.

## KCI 저널 게재 목표

- 일반 KCI 이공계 저널 기준(특정 학회지 미확정)으로 재현성·통계적 엄밀성·관련연구를
  보완 중 — 상세 계획은 `docs/kci-submission-gap-analysis.md` 참고.
