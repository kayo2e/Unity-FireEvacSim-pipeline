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

**10. PPO 하이퍼파라미터 문헌 대조 감사 (Phase 5c, 진행 중)**: Andrychowicz et
al. (2020)이 연속제어에서 엔트로피 보너스의 이득 근거를 찾지 못했다고 보고한
점에 착안, `ent_coef=0.05`(기존 기본값) vs `0.0`을 같은 조건(40만 스텝,
`--max-scenario 4`, fresh 학습)으로 비교 중. ent_coef=0.0 결과: S4 생존율
59.6±34.5%(n=30). ent_coef=0.05 fresh 컨트롤은 아직 실행/비교 전이라 결론
보류(공정 비교를 위해 두 값 모두 같은 40만 스텝 기준이어야 한다. production
모델의 66.0±31.2%는 3,500,000스텝으로 학습량이 달라 직접 비교 불가).

### Next

- Phase 5c 마무리: ent_coef=0.05 fresh 컨트롤 실행 결과 확인 → 두 fresh 결과
  비교 → production 모델(`fire_evac_model_40ppl.zip`)이 실험용 저장으로
  덮어써지지 않았는지 재확인(백업: `/tmp/fireevac_model_backup/`).
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
