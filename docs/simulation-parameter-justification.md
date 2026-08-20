# 시뮬레이션 파라미터 근거표

`raw 데이터셋이 없다`는 한계를 KCI 심사에서 방어하는 표준적인 방법은, 군중·화재 물리
파라미터가 **실측 데이터로 검증된 문헌값에서 나왔는지**를 명시하는 것이다. 이 문서는
`stage2/env_core.py`의 모든 물리 상수를 그 근거(직접 인용 / 정성적 영감 / 구현상
설계값)와 함께 정리한다. 세 카테고리로 구분한 이유는, "문헌값을 그대로 썼다"와 "문헌의
현상을 시뮬레이션에 맞게 재구성했다"를 섞어서 주장하면 심사자가 검증하려 할 때 바로
드러나는 약점이 되기 때문이다. 정직하게 구분해두는 편이 오히려 방어에 유리하다.

## 1. 문헌값에서 직접 도출 (Direct)

| 상수 | 코드 값 | 문헌 근거 | 대응 관계 |
|---|---|---|---|
| `DENSITY_SLOW_MAX` | 0.80 | Fruin (1971), *Pedestrian Planning and Design* | Fruin의 보행자 LOS(Level of Service) 속도-밀도 곡선: 자유 흐름 속도(~80 m/min, LOS A)에서 혼잡 밀도(4명/m² 초과, LOS F)로 갈 때 속도가 ~15–20 m/min까지 떨어진다. 약 75~81% 감소 구간과 일치하고, 최대 감소율 80%는 이 범위 중간값을 취한 것 |
| `EXIT_CAPACITY` (학습 시 1) | 1 (완화 데모 시 2) | Henderson (1974), 보행자 출구 처리율 문헌 | 출구 하나가 스텝당 처리 가능한 인원을 1~2명으로 제한. 병목 실험(bottleneck simulation) 설계와 직접 연결됨 |

## 2. 문헌 현상에서 정성적으로 재구성 (Qualitative-inspired)

문헌이 "이런 현상이 존재한다"는 것은 실측으로 뒷받침하지만, 구체적 수치(거리 단위,
확률값)는 이 시뮬레이션의 그리드 스케일에 맞춰 새로 정한 것이다. 문헌이 그 정확한
숫자를 보고하지는 않는다.

| 상수 | 코드 값 | 문헌 근거 (정성적) | 실제로 문헌이 뒷받침하는 것 / 뒷받침하지 않는 것 |
|---|---|---|---|
| `PANIC_FIRE_DIST` | 25.0 (셀) | Helbing, Farkas & Vicsek (2000), *Nature* | 뒷받침: 위험 근접도가 공황 수준에 영향을 준다는 정성적 메커니즘. 뒷받침 안 함: "25셀"이라는 구체적 임계값(이 프로젝트 그리드 40×25의 스케일에 맞춘 설계값) |
| `PANIC_RANDOM_MAX` | 0.40 | Helbing et al. (2000), panic 하에서 "faster-is-slower", 비합리적 군집 이동 | 뒷받침: 공황이 클수록 유도 신호를 무시하고 비합리적으로 움직이는 경향. 뒷받침 안 함: 정확히 40%라는 확률값 |
| panic 상승/하강 비대칭 (`alpha=0.3` 상승 / `0.05` 하강) | 설계값 | 공황·스트레스 반응의 일반적 심리학적 패턴(급상승·완만한 하강) | 뒷받침: 비대칭 동역학의 방향성. 뒷받침 안 함: 정확한 시상수(time constant) |

## 3. 순수 구현 설계값 (Design choice, no literature claim)

문헌 인용을 붙이지 않는 편이 정직한 값들이다. 그리드 해상도, 관측 반경 등 시뮬레이션
엔지니어링 상의 선택.

| 상수 | 코드 값 | 성격 |
|---|---|---|
| `DENSITY_RADIUS` | 1 (체비쇼프 거리, 8-이웃) | 그리드 해상도에 따른 로컬 밀도 측정 반경. Fruin의 "명목 면적(m²/인)" 개념을 셀 단위로 근사하되, 정확한 환산식은 아님 |
| `QUEUE_RADIUS` | 4 | F7/F8(출구 근접 혼잡도) 피처의 BFS 측정 반경. 순수 관측 설계 파라미터 |
| `CELL_CAPACITY` | 1 (완화 데모 시 2) | 셀당 동시 점유 인원 제한. `EXIT_CAPACITY`와 함께 병목 실험을 위한 설계값 |

## 결론: raw 데이터 없이도 방어 가능한 이유, 그리고 남은 숙제

1번 카테고리(Fruin/Henderson 직접 대응)는 이미 실측 데이터로 검증된 값을 그대로
가져온 것이라 추가 조치가 필요 없다. 2번 카테고리(Helbing 정성적 재구성)가 실제
심사에서 지적받을 가능성이 가장 높은 지점이다. 대응 방법은 두 가지다.

- **민감도 분석(sensitivity analysis)**: `PANIC_FIRE_DIST`·`PANIC_RANDOM_MAX`를
  ±30~50% 범위에서 흔들어 표 3 결과(생존율·완료 시간)가 질적으로 안정적인지 확인.
  결과가 파라미터 선택에 민감하지 않다는 것을 보이면, 정확한 실측값이 없어도
  결론의 강건성(robustness)을 주장할 수 있다. GPU 없이 로컬 CPU로도 수행 가능하다.
- 3번 카테고리(순수 설계값)는 "문헌 근거"라고 주장하지 않고 "시뮬레이션 설계
  선택"이라고 명시하는 것 자체가 정직한 대응이다. 이 문서가 이미 그 역할을 한다.

## "raw 데이터셋이 아예 없어도 되는가?": 실은 있다

> **상태: 교수님 논의 대기.** 어떤 서브데이터셋이 이 프로젝트의 병목 구조와
> 가장 가까운지, 검증 절을 논문에 어느 정도 비중으로 넣을지는 지도교수님과
> 상의 후 착수한다. 아래는 논의를 위한 조사 내용이며, 실제 다운로드·비교 작업은
> 아직 시작하지 않았다.

화재 상황 자체의 raw 데이터(실제 화재 대피 영상·센서 로그)는 윤리적으로 수집이
불가능하지만, **이 프로젝트가 이미 인용 중인 군중 물리 모델(Fruin, Helbing)을
검증하는 데 쓸 수 있는 실측 보행자 데이터는 실제로 공개돼 있다.** raw 데이터가
"전혀 없다"는 전제 자체가 정확하지 않다.

- **[Pedestrian Dynamics Data Archive](https://ped.fz-juelich.de/extda)**
  (Forschungszentrum Jülich, CC BY-SA 4.0): 통제된 실험실 조건에서 촬영한 실제
  보행자 이동 영상 + PeTrack으로 추출한 정밀 궤적 데이터. 밀도·속도·흐름을 임의
  시점·위치에서 계산할 수 있어, `DENSITY_SLOW_MAX=0.80`(밀도-속도 감소 곡선)을
  Fruin의 1971년 수치가 아니라 **최신 실측 데이터로 직접 재검증**하는 데 쓸 수 있다.
  무료 공개, 별도 승인 절차 없음.
- Xie, F. et al. (2025). *Dense Crowd Dynamics and Pedestrian Trajectories: A
  Multiscale Field Dataset from the Festival of Lights in Lyon*. **Scientific
  Data** (Nature). 실제 대규모 행사에서 촬영한 밀집 군중 궤적 데이터다. 화재
  상황은 아니지만 고밀도 병목 상황(EXIT_CAPACITY/CELL_CAPACITY 병목 실험과
  직접 관련)의 실측 검증에 쓸 수 있다.

**결론**: "raw 데이터가 없다"가 아니라 **"화재 데이터는 없지만, 그 화재
시뮬레이션이 의존하는 군중 물리 모델을 검증할 실측 데이터는 있다"**로 정정해야
한다. 다음 단계로 제안: Jülich 아카이브에서 `DENSITY_RADIUS`/`DENSITY_SLOW_MAX`에
대응하는 밀도-속도 실험 데이터셋 1~2개를 받아, 이 프로젝트의 `_effective_speed()`
함수가 실측 곡선과 얼마나 가까운지 비교하는 절을 논문에 추가한다. raw 데이터
부재를 정면으로 반박하는 가장 강력한 근거가 된다.

## 참고 문헌

- Fruin, J. J. (1971). *Pedestrian Planning and Design*. Metropolitan Association of Urban Designers.
- Helbing, D., Farkas, I., & Vicsek, T. (2000). Simulating dynamical features of escape panic. *Nature*, 407, 487–490.
- Henderson, L. F. (1974). On the fluid mechanics of human crowd motion. *Transportation Research*, 8(6), 509–515.
- Forschungszentrum Jülich, IAS-7. *Pedestrian Dynamics Data Archive*. https://ped.fz-juelich.de/extda (CC BY-SA 4.0).
- Xie, F. et al. (2025). Dense Crowd Dynamics and Pedestrian Trajectories: A Multiscale Field Dataset from the Festival of Lights in Lyon. *Scientific Data*, 12.
