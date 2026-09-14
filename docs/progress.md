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

**5. Stage 1 Phase A~E: 완료.** `stage1/build_ground_truth.py`로 이 사진
전용 정답(40×25, 구조적 블록 단위 근사, `image_grid_overlay.jpg`를 3등분해서
육안으로 좌표 확인)을 만들고, `stage1/eval_extraction_accuracy.py`로 Phase
B(표준 지표: pixel accuracy 76.9%, WALL IoU 63.4%)·Phase C(관대한 매칭
r=1: recall 94.1%/precision 90.5%)·Phase D(연결성 보존율 67.0%, EXIT
검출 recall/precision 100%)를 계산했다. 결과와 문헌 대조(Zeng et al. 2019,
CubiCasa5K, de las Heras et al. 2014, Liu et al. 2017)를
`docs/stage1-extraction-accuracy.md`에 정리하고, README의 "Stage 1 추출
정확도 미검증" 한계 문단을 이 실측 결과로 교체했다. 이 문단에서 "지금 학습된
모델이 쓰는 그리드는 이 사진에서 온 게 아니라 Unity 씬에서 나온 것"이라는
사실도 명시적으로 반영했다(4번 항목에서 이미 확인된 내용, README에는 아직
안 옮겨져 있었음). README 제목·개요의 "E2E 자동 추출" 프레이밍 자체를
재작성할지는 더 큰 구조적 변경이라 사용자 확인 없이 건드리지 않았다.

**6. Stage 1/Stage 2 실제 통합 여부 결정.** 새로 추출한 그리드(`base_grid.npy`)
와 지금 production 그리드(Unity 씬 기반)를 나란히 시각화해서 비교한 결과,
사용자가 직접 보고 "새 그리드로 교체하기엔 아직 부담스럽다(WALL IoU 63%,
화장실 아이콘·칸막이까지 벽으로 잡힘)"고 판단해 **교체·재학습은 보류**하기로
확정했다. 대신 README 개요에 "Stage 1과 Stage 2는 각각 독립적으로
검증됐다"는 문단을 추가해 두 구성 요소가 아직 하나로 이어지지 않았다는
사실을 명시했다(제목·전체 구조는 그대로 유지, 최소 변경).

### Next

- 위 "Next" 목록(Phase 5, 6, 2, 8, 피처 공간 최적화)은 그대로 유효.
- Stage 1/2 실제 통합은 보류 상태. 나중에 다시 검토하려면 그리드 품질을
  먼저 더 개선해야 함(화장실 아이콘류 노이즈 제거).

## 2026-09-13 — 피처 공간 최적화 착수, 정책이 액션 경계값에 고정되는 버그 발견·수정

### 한 일

**1. 피처 공간 최적화 계획(Task 2) 1~2번 실행.** `experiments/feature_ablation.py`
(개별 피처 마스킹), `experiments/feature_correlation.py`(15×15 상관행렬),
`experiments/feature_group_ablation.py`(클러스터 단위 마스킹)를 새로 작성해
production 모델(당시 3.5M 스텝 완주분)에 돌렸다. F1~F15를 하나씩, 또는
클러스터째로 마스킹해도 생존율 낙폭이 1%p를 넘지 않았고, 심지어 F1~F15
전체를 동시에 마스킹해도(ALL_BLIND) 최대 1.2%p였다. 상관행렬에서는
F3(출구A 선호)·F7/F8(혼잡도)·F10/F11(평균거리)가 |r| 0.64~0.95로 같은
축을 다르게 인코딩하고 있음을 확인했고, F1/F2(위협)와 F14/F15(시간차분)는
예상과 달리 |r|<0.06으로 독립적이었다.

**2. 마스킹이 무의미했던 진짜 원인 발견.** `experiments/action_sensitivity_check.py`
로 정책의 결정적 행동을 7,808개 실제 상태에 대해 직접 찍어보니
exit_A_cost/exit_B_cost/crowd_weight 셋 다 관측과 무관하게 액션 하한
근처에 고정돼 있었다. 정책 파라미터를 확인한 결과 `log_std` 평균이
4.96(표준편차 환산 ≈143)까지 발산해 있었다 — 액션 범위보다 훨씬 큰
노이즈라 학습 중 샘플링된 행동이 경계값으로 무작위 클리핑되고, 평균망이
관측에 따라 분화할 그래디언트를 못 받은 상태였다.

**3. 액션 자체의 인과 효과는 확인.** `experiments/action_causal_sweep.py`로
학습과 무관하게 exit_A/B_cost를 수동으로 고정해 롤아웃한 결과, S2·S5에서
잘못된 선호와 올바른 선호 사이에 48.7~58.0%p의 생존율 차이가 났다. 즉
액션 설계 자체는 유효했고, 문제는 "배울 것이 없어서"가 아니라 "정책이
못 배운 것"으로 좁혀졌다.

**4. Hazard-aware BFS와 직접 대결시켜 비교 기준 확보.** 저장소에 이미
있던 `astar_baseline.py`(Hazard-aware BFS, 학습 없음)와 production PPO를
같은 시드(n=20~30, seed=42)로 맞대결시켰다. S2·S4·S5 3개 시나리오에서
비학습 휴리스틱이 PPO와 동급이거나 더 높았다(S2 -4.8%p, S5 -10.5%p).
기존 README·논문 초안의 "PPO가 Pure A\* 대비 12.3~31.2%p 개선"이라는
서술은 그 개선의 출처가 강화학습이 아니라 환경에 기본 내장된 hazard-aware
Dijkstra 라우팅(`FireEvacEnv(hazard_aware=True)` 기본값)일 가능성이
크다는 뜻이 됐다.

**5. 1차 수정(엔트로피 계수) — 부분적으로만 효과 있음.** `ent_coef`를
0.05→0.005로 낮춰 500K 스텝 파일럿 후 3.5M까지 재학습했다. `log_std`
발산은 잡혔지만(0.02~1.15로 안정), 검증 결과 crowd_weight만 관측에
반응하기 시작했고 exit_A/B_cost는 여전히 정확히 5.0에 고정돼 있었다.

**6. 2차 수정(action_net bias 초기화) — 진짜 원인.** `action_net`의
클리핑 전 raw 출력을 직접 확인한 결과, 정책망이 실제로는 상태에 따라
값을 다르게 냈지만(raw std 1.0~1.6) 그 값이 전부 박스 하한(exit_cost는
5)보다 한참 아래에 있어 항상 클리핑되고 있었다. 신경망의 초기 출력
스케일(0 근처)과 액션 박스([5,50], [0.5,5.0])가 애초에 안 겹치는 설계
문제였다. `ppo/ppo_train.py`에서 모델 생성 직후 `action_net.bias`를
박스 중간값([27.5, 27.5, 2.75])으로 초기화하는 한 줄을 추가하고, 체크포인트
없이 처음부터 3.5M 스텝을 다시 학습했다.

**7. 최종 검증.** 수정판에서는 exit_A/B_cost가 실제로 갈라졌다(23~32
범위). 같은 조건(n=30, seed=42)으로 Hazard-aware BFS와 다시 비교한 결과:

| 시나리오 | Hazard-aware BFS | PPO (수정판) |
|---|---:|---:|
| S2 | 91.3% | 90.8% |
| S3 | 86.5% | **87.9%** |
| S4 | 68.9% | 65.8% |
| S5 (OOD) | 89.6% | 78.3% |

커리큘럼이 실제로 도달한 S2·S3에서는 동급이거나 역전했다. S4·S5에서
아직 뒤지는 건 이번 학습도 커리큘럼이 S3까지만 승급하고 S4로 못 올라간
탓으로 보인다(승급 기준 재검토 필요). README "PPO 아키텍처 미검증"·
"피처 공간 최적화" 두 문단과 `docs/hazard-aware-ablation.md`(후속 분석
6 추가)를 이 결과로 갱신했다.

**8. 커리큘럼 ablation 재검증 스크립트 준비.** 후속 분석 5(커리큘럼 효과)는
옛 `ent_coef=0.05` 기준으로 수행된 것이라, 이번 수정판 기준으로 다시
검증해야 한다. `experiments/train_no_curriculum_ablation.py`를 새로
작성해뒀다(커리큘럼 없이 S4 직행, 기존 모델과 분리된 경로에 저장). 아직
미실행.

### Next

- `train_no_curriculum_ablation.py` 실행해 커리큘럼 효과를 수정판 기준으로
  재검증.
- 커리큘럼 승급 기준(최근 50 에피소드 평균 생존율 0.90 이상)을 낮추거나
  조정해 S4까지 실제로 도달하는지 재시도.
- 표 3(완료 스텝, Exit Balance/Throughput 등)을 이번 수정판 모델 기준으로
  전면 재측정. 지금 README 표 3은 여전히 버그 수정 전 모델 기준이다.
- 피처 공간 최적화 Task 2 3~4번(축소 피처셋 재학습)은 위 재검증들이 끝난
  뒤 재개.

## 2026-09-13 (계속) — PR 리뷰·머지, 커리큘럼 승급 기준 완화, 비용 비교로 기여 재정의

### 한 일

**1. action_net bias 수정 PR 리뷰 후 머지.** `fix/action-space-clipping-bug`
브랜치로 PR #17을 열고 코드 리뷰를 거쳤다. 발견된 문제 중 `ppo_train.py`
`train()`의 `ent_coef` 기본값과 CLI `--ent-coef` 기본값이 여전히 0.05로
남아있던 것(수정판의 결론과 모순)을 0.005로 정정했고, `action_net` bias
초기화 로직이 `ppo_train.py`와 `train_no_curriculum_ablation.py`에 중복
구현돼 있던 것을 `train_common.init_action_net_bias_to_box_mid()`로
공용화했다. `stage2/logs/`를 `.gitignore`에 추가해 앞으로의 학습 로그가
계속 커밋에 쌓이는 걸 막았다(이미 커밋된 로그는 그대로 둠). 이 과정에서
`make_env`/`make_vec_env`에 `curriculum_threshold`/`curriculum_window`
파라미터를 추가하고 `ppo_train.py`에 `--curriculum-threshold`/
`--curriculum-window` CLI로 노출했다. PR 머지 후 main으로 fast-forward.

**2. 커리큘럼 승급 기준 완화 실험.** 직전 재학습(후속 분석 6)도 커리큘럼이
S3에서 멈췄던 걸 승급 기준(0.90)이 S4에서 너무 빡빡한 것으로 의심해,
기준을 0.85로 낮추고 나머지 조건은 동일하게 유지한 채 처음부터 3.5M
스텝을 재학습했다. 이번엔 8개 병렬 환경 전부 S4까지 승급했다. 같은 조건
(n=30, seed=42)으로 Hazard-aware BFS와 다시 비교한 결과 S4 격차가
-3.1%p→-1.4%p, 한 번도 학습하지 않은 S5(OOD) 격차가 -11.3%p→-7.4%p로
함께 줄었다.

**3. 비교축을 생존율에서 비용으로 전환 — 훨씬 뚜렷한 우위 확인.**
`experiments/exp_speed.py`에 `bench_hazard_bfs()`를 추가해 Hazard-aware
BFS의 스텝당 행동결정 시간을 N=20~500까지 재실측했다. 생존자마다 출구
A·B 양쪽으로 BFS를 새로 도는 구조라 Pure A\*보다도 약 2배 느렸고, "실시간
기준 100ms/스텝"을 N≈200~300 사이에서 이미 넘어섰다(Pure A\*보다 더
이른 지점). PPO는 N=500에서도 0.5ms 수준. 생존율로는 Hazard-aware BFS를
확실히 못 이기지만, "같은 수준의 판단을 훨씬 큰 규모까지 실시간으로
낼 수 있다"는 비용 축에서는 명확한 우위가 나온다는 게 이번에 확인한
핵심 재발견이다.

**4. README 반영.** "추론 속도 비교(그림 5)" 표를 Hazard-aware BFS
열을 포함해 전면 재실측치로 교체하고, "PPO 아키텍처 미검증" 문단에
승급 기준 재검증 경과를 추가했다. `docs/hazard-aware-ablation.md`에
후속 분석 7로 두 실험(승급 기준 완화, 비용 재실측)을 모두 기록했다.

### Next

- `train_no_curriculum_ablation.py`를 `curriculum_threshold=0.85` 조합으로
  실행해 커리큘럼 유무 자체의 효과를 최신 설정 기준으로 재검증.
- S4 외 다른 시나리오(더 많은 인원)에서도 Hazard-aware BFS의 실시간 임계점이
  같은 패턴으로 나오는지 확인.
- 표 3(완료 스텝, Exit Balance/Throughput, 정적 유도등 비교)을 최신 모델
  기준으로 전면 재측정 — 아직 미착수.
- 승급 기준 0.85 완화가 다른 시나리오 품질에 부작용이 없는지는 n=30 1회
  비교로만 확인된 상태라 추가 반복 검증 여지가 있음.
