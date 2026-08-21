# 진행 로그

## 2026-08-20 — hazard_aware 버그 수정, 표3 전면 재측정, README 논문 형식 재구성

### 한 일

**1. hazard_aware 버그 발견·수정**
"화재 무시"가 전제인 베이스라인(Simple A\*, Pure A\*, 정적 유도등) 3개가 실제로는
`FireEvacEnv._compute_bfs_with_risk()`에서 화재 회피 비용을 무조건 적용받고
있었다. `env_core.py`에 `hazard_aware: bool` 파라미터를 추가해 고쳤다(기본값
`True`로 기존 PPO/Hazard-aware A\*는 그대로 보존). 상세: `docs/hazard-aware-ablation.md`.

**2. 표 3 전면 재측정** (S1~S5, n=30, seed=42 페어링, paired t-test)

| 시나리오 | 정적 유도등 | A\* | PPO | PPO vs A\* |
|---|---:|---:|---:|---|
| S1 | 99.8±0.9 | 99.8±0.9 | 99.8±0.9 | 검정 불성립(ceiling) |
| S2 | 59.0±16.8 | 54.4±22.0 | 85.7±6.1 | p<0.0001 (+31.2%p) |
| S3 | 57.8±12.1 | 60.5±14.3 | 88.1±6.3 | p<0.0001 (+27.6%p) |
| S4 | 50.0±25.8 | 53.0±29.1 | 66.0±31.2 | p=0.0008 (+13.0%p) |
| S5 (OOD) | 63.1±15.4 | 67.1±20.5 | 79.4±8.8 | p=0.0006 (+12.3%p) |

버그 수정 전에는 S4가 "생존율 동률"로 잘못 나왔었다.

**3. Exit Balance/Throughput 재해석**: "PPO가 F7/F8로 출구를 균등 분산시킨다"는
기존 README·docstring 서술이 실측과 반대로 나왔다. PPO의 Exit Balance는 A\*보다
항상 낮고(S2: 0.226 vs 0.411), Throughput은 항상 높다(S2: 0.394 vs 0.302). "균등
분산"이 아니라 "더 나은 쪽으로 결단력 있게 몰아 처리 효율을 높이는" 전략으로
README 3곳(표3 해설·F7/F8 각주·보상함수 각주) 정정.

**4. N 스케일링(밀도 일반화 경계, Phase 4b)**: 커리큘럼 최대 인원(40명)을 훨씬
넘는 N=40~500까지 재측정. PPO 우위는 유지되지만 폭이 꾸준히 줄어든다
(N=40: +15.6%p → N=200: +7.4%p). 사용자가 이전에 관찰한 "N=200에서 A\*가 역전"은
hazard_aware 버그 수정 이전 관찰이었음을 확인.

**5. 추론속도 재실측(Phase 4)**: N=20/50/100/150/200/300/500 7단계로 실측.
PPO는 O(1)(~0.2ms 고정), A\*는 O(N)(N당 ~0.35ms). 기존 README의 "N=200 → A\*
~2,000ms" 주장이 실측(53.6ms)과 37배 차이 나는 근거 없는 추정치였음을 발견,
정정.

**6. 정적 유도등(Static Signage) 베이스라인 추가**: 이 분야 표준 비교군(최초 1회
계산한 경로를 화재와 무관하게 고정)을 `stage2/baselines/static_signage_baseline.py`로
신설, 표 3에 반영.

**7. README를 논문/보고서 형식으로 재구성**: 개요 → 관련 연구 → 방법론 → 실험
결과 → 시각화 → 설치 → 파일 구조 → 한계 및 향후 연구 → 참고문헌 순서로 재배치.
S1~S5 통합 비교 GIF로 시각화 절 교체. em-dash(" — ") 부연설명 구문을 전부
제거하고 문장 종결/콜론으로 정리(AI가 쓴 것처럼 읽히는 문체 지적에 따른 수정).
같은 정리를 `docs/kci-submission-gap-analysis.md`에도 적용.

**8. `CLAUDE.md`를 저장소에서 삭제** (사용자 요청).

**9. 피처 공간(F1~F15) 최적화 조사 + 계획 문서화**: 현재 15차원 관측의 각
피처별 계산 방법·용도를 코드 기준으로 인벤토리화하고, 잠재적 중복 조합
(F1/F2 vs F14/F15, F3 vs F10/F11, F4 vs F5)을 식별했다. Kim & Ha(2020,
observation-space 최적화)와 최근 군중 대피 MARL 연구의 localized observation
경향을 근거로, 이게 이 연구의 명시적 한계임을 README "한계 및 향후 연구"에
반영했다. 실행 계획(permutation 중요도 측정 → 상관관계 분석 → 축소 피처셋
재학습 비교)은 `docs/feature-space-optimization-plan.md`에 정리만 해두고
**아직 실행하지 않았다**.

**10. PPO 하이퍼파라미터 문헌 대조 감사 (Phase 5c): 완료**. Andrychowicz et
al. (2020)이 연속제어에서 엔트로피 보너스의 이득 근거를 찾지 못했다고 보고한
점에 착안, `ent_coef=0.05`(기존 기본값) vs `0.0`을 같은 조건(40만 스텝,
`--max-scenario 4`, fresh 학습)으로 비교했다. 결과: ent_coef=0.0 S4 생존율
59.6±34.5%, ent_coef=0.05 S4 생존율 65.4±28.2%(둘 다 n=30). 같은 스텝 수
기준으로 0.05가 오히려 5.8%p 높지만 Welch's t-test 근사 검정으로는 유의하지
않다(t≈0.71). "엔트로피를 낮추면 개선된다"는 가설은 지지되지 않아 기존
기본값 `ent_coef=0.05`를 그대로 유지하기로 했다. production 모델은 실험
중 두 차례 임시로 덮어써졌으나 매번 `/tmp/fireevac_model_backup/`에서
즉시 복원해 최종적으로는 원상태다. 상세: `docs/hazard-aware-ablation.md`
후속 분석 4.

### Next

- Phase 5b: 커리큘럼 학습 유무 ablation (같은 스텝 수·시드로 비교).
- Phase 5: PPO에 경량 모듈(Mamba류 지역성 요약 레이어) 추가 실험.
- Phase 6: Stage 1 그리드 추출 정확도·강건성 정량화.
- Phase 2: 새 평면도 1~2개로 zero-shot 일반화 검증. 사용자가 새 평면도
  이미지를 제공해야 진행 가능.
- Phase 8: 그리드-실측 스케일(㎡/셀) 확정, 소방법령 수용인원 기준 대조.
  "상상관 2층" 실제 면적 확인 필요.
- 피처 공간 최적화(위 9번): `docs/feature-space-optimization-plan.md`의 task2
  계획 실행. 아직 착수 전.
- `feat/n-scaling-breakdown` 계열 잔여 작업이 정리되면 main과 완전히 합류
  확인.

## 2026-08-21 — 커리큘럼 효과 검증(Phase 5b), README 한계 보강

### 한 일

**1. README 한계 절 보강**: 카톡 초안을 계기로 대기큐 항목 중 README에 아직
반영 안 된 3개를 "한계 및 향후 연구"에 추가했다. Stage 1 추출 정확도 미검증,
인원수(N) 설정 근거 미확정(건물 실측 면적 필요), PPO 아키텍처·커리큘럼 설계
미검증.

**2. Phase 5b: 커리큘럼 학습 효과 검증. 완료**. `train_common.py`에
`start_scenario` 파라미터를 추가해(`ppo_train.py`의 `--start-scenario` CLI로
노출) 커리큘럼 진급 없이 특정 시나리오로 처음부터 학습할 수 있게 했다. 같은
조건(40만 스텝, 40명, ent_coef=0.05, fresh)에서 "커리큘럼 없이 S4 고정"과
"커리큘럼(S1→S4)"을 비교했다.

| 설정 | S4 생존율(n=30) |
|---|---:|
| 커리큘럼 없음 | 53.3±36.6% |
| 커리큘럼 있음 | 65.4±28.2% |

+12.15%p 차이지만 Welch's t-test 근사로는 유의하지 않다(t≈1.44, n=30
과소검정력 가능성). 다만 분산이 커리큘럼 쪽이 훨씬 작아(36.6%→28.2%) 더
안정적으로 수렴한다는 방향성은 확인됐다. 기존 커리큘럼 설계를 유지하기로
했다. production 모델은 실험 중 임시로 덮어써졌다가 즉시 백업에서 복원해
최종적으로는 원상태다. 상세: `docs/hazard-aware-ablation.md` 후속 분석 5.

### Next

- Phase 5: PPO에 경량 모듈(Mamba류 지역성 요약 레이어) 추가 실험.
- Phase 6: Stage 1 그리드 추출 정확도·강건성 정량화.
- Phase 2: 새 평면도 1~2개로 zero-shot 일반화 검증. 사용자가 새 평면도
  이미지를 제공해야 진행 가능.
- Phase 8: 그리드-실측 스케일(㎡/셀) 확정, 소방법령 수용인원 기준 대조.
  "상상관 2층" 실제 면적 확인 필요.
- 피처 공간 최적화: `docs/feature-space-optimization-plan.md`의 task2 계획
  실행. 아직 착수 전.

## 2026-08-21 — 저장소 위생 정리, 커밋 히스토리 영어 재작성, Stage 1 정밀도 계획 승인

### 한 일

**1. GitHub Contributors에서 Claude 재등장 문제 해결**: 리포 사이드바 Contributors
위젯이 저장소 전체 브랜치를 긁어 집계한다는 걸 확인. `main`·오늘 작업한 브랜치는
전부 깨끗했고, 원인은 병합된 적 없는 `eval/comparison`(팀원 주요셉 계정, 자신의
Claude Code 사용으로 생긴 `Co-Authored-By: Claude` 트레일러 5건 포함) 브랜치였다.
`eval/comparison` 삭제 + 이미 main에 병합된 leftover 브랜치 15개도 함께 정리(18개
→ main 1개). `gh api .../contributors`로 kayo2e 1명만 남은 것 확인. GitHub의
Contributors 위젯 자체는 캐시가 오래 남아 화면에는 지연 반영될 수 있음(데이터는
이미 정상).

**2. 커밋 히스토리 전체를 영어로 재작성**: 기존 `CLAUDE.md`(삭제 전)는 "한국어
관례 유지"였으나 사용자가 이번에 전체 영어 통일을 명시적으로 요청. 초기 66개
커밋(한국어)을 `git-filter-repo` 커밋 콜백으로 번역해 재작성, 나머지 50개는 이미
영어라 그대로 유지. Authorship·내용(트리)은 그대로, 메시지만 교체. `--force-with-lease`
로 재푸시 완료. 재작성 후에도 Contributors·브랜치 상태 정상 확인.

**3. Stage 1 벽 마스크 정밀도 개선 + 평가 계획 수립(Plan Mode)**: `base_grid_vis.jpg`
육안 검토에서 벽 이중선·홀 결함 확인 → 원인을 `binary_img.jpg` 픽셀 단위로 직접
진단(원본 도면이 벽 두께를 평행한 두 선으로 그리는 제도 관례, 해칭 도형 내부가
임계값 미달로 안 채워짐). 문헌 조사(Zeng et al. 2019, CubiCasa5K, de las Heras et
al. 2014, Liu et al. 2017 Raster-to-Vector, SLAM occupancy-grid 비교 문헌) 기반으로
Phase 0(정밀도 개선: morphological closing + hole filling) → Phase A(이 사진 전용
ground truth 제작) → Phase B/C/D(표준 IoU/관대한 매칭/연결성 보존율 평가) → Phase
E(문서화) 계획을 세워 사용자 승인받음. 계획 파일:
`~/.claude/plans/whimsical-wandering-stroustrup.md`.

**4. Stage 1 Phase 0: 벽 마스크 정밀도 개선, 대부분 완료.** 원래 계획한 "이중선
벽 닫힘 연산"은 실제로 시도해보니 위험한 접근으로 판명됐다. 실측된 이중선 간격
(24~26px)에 맞춘 29px 닫힘 커널을 적용하자 출구 하나가 완전히 고립되는 걸 확인
했다(그 부분이 실제로는 벽 결함이 아니라 진짜 복도 폭이었을 가능성이 높음). 이
접근은 보류하고, 대신 연결성 자체를 정량 확인하는 BFS 체크
(`env_core.verify_connectivity()`와 동일 로직)를 새로 만들어 돌려본 결과, 닫힘
연산 없이도 이미 원본 파이프라인 자체가 55%(354/641칸)를 어느 출구에도 도달
불가능한 상태로 만들고 있었다는 훨씬 근본적인 문제를 발견했다.

  1. **건물 외곽 밖 여백**이 그냥 HALL로 남아있던 게 최대 원인(도달 불가 셀의
     대부분). 외곽 컨투어(`cv2.findContours` RETR_EXTERNAL)로 건물 폴리곤을
     찾아 그 밖을 전부 WALL로 채우는 `_mask_outside_building()`을 추가해
     341→130 unreachable로 개선했다.
  2. 남은 130칸은 201~209호 각 방이 복도로 통하는 **문(door) 기호**가 안
     뚫려서 고립된 것으로 확인했다(원본 도면에서 문이 얇은 직사각형 외곽선으로,
     셀 하나보다 작게 그려져 있어 5% 풀링 임계값에서 항상 WALL로 밀림). 이 문
     기호 하나를 템플릿으로 잘라 `cv2.matchTemplate`으로 같은 벽선을 따라 11개
     문을 모두 찾아(`find_door_openings()`) 해당 셀을 강제로 HALL 처리해
     130→**8** unreachable(428칸 중 98.1% 연결)까지 개선했다. 이 문 검출은
     이 도면 고유의 그리기 방식에 맞춘 템플릿 매칭이라, 초록 출구 아이콘(표준
     소방 색상) 검출과 달리 다른 건물 사진에는 그대로 일반화되지 않는다. 문서화
     해둘 한계다.
  3. 남은 8칸은 작은 고립 포켓(화장실 칸막이 등으로 추정) 수 개뿐이라 여기서
     멈췄다. 완전 100% 연결은 추가 조사가 필요하면 나중 과제로 남긴다.

  산출물: `stage1/build_base_grid.py`(정밀도 개선 반영), `stage1/door_template.png`
  (문 기호 템플릿), `stage1/base_grid_connectivity.jpg`(BFS 도달 불가 셀 빨간색
  시각화, 디버깅용).

### Next

- Stage 1 Phase A(이 사진 전용 ground truth 제작)부터 이어서. Phase 0는 위
  4번 항목으로 완료 처리.
- 위 "Next" 목록(Phase 5, 6, 2, 8, 피처 공간 최적화)은 그대로 유효.
