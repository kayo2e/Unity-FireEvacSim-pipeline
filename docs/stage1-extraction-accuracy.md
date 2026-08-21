# Stage 1 추출 정확도 평가

`stage1/build_base_grid.py`(사진 → 40×25 그리드 자동 추출)의 결과를 이 사진
한 장에 대해 직접 만든 정답(ground truth)과 비교한 결과다. 배경과 방법론은
[Stage 1 벽 마스크 정밀도 개선 + 평가 계획](../docs/progress.md)(2026-08-21)
참고. 재학습·GPU 불필요, 전부 CPU + 수작업 정답 제작 1회로 끝났다.

## 왜 기존 `BASE_GRID`가 아니라 새 정답을 만들었는가

`env_core.py`의 `BASE_GRID`(실제 학습에 쓰인 40×25 그리드)는 이 사진에서 나온
게 아니라 Unity 에디터에서 수작업으로 만든 3D 씬을 내보낸 것이다(코드 주석
"Unity 원본 인코딩에서 변환" 확인). 두 그리드를 겹쳐보면 구조 자체가 다르다
(WALL IoU 19%). 그래서 이 사진 하나에 대한 정답을 `stage1/build_ground_truth.py`
로 직접 만들었다: 40×25 그리드를 사진 위에 겹친 오버레이(`image_grid_overlay.jpg`)
를 3등분해서 방/복도/코어 경계를 육안으로 읽고 좌표로 옮겼다. ROOM/HALL은
구분하지 않는다. `env_core.py`에서 `WALKABLE = {HALL, EXIT, ROOM}`로 셋을
동일하게 취급해 구분할 실익이 없다.

이 정답은 셀 단위 완전 정밀 라벨링이 아니라 구조적 블록(방/복도/코어) 단위
근사다. 40×25 해상도에서는 문 폭 같은 미세 구조가 셀 하나보다 작아서, 사람이
직접 봐도 셀 경계에서 판단이 갈리는 지점이 있다. 이건 정답 자체의 한계로
남겨둔다.

## Phase 0: 벽 마스크 정밀도 개선, 연결성 45% → 98.1%

애초 계획했던 "이중선 벽 닫힘 연산"은 실측된 이중선 간격(24~26px)에 맞춘
29px 닫힘 커널을 적용하자 출구 하나를 완전히 고립시키는 걸 확인해 보류했다
(그 부분이 벽 결함이 아니라 진짜 복도 폭이었을 가능성이 높음). 대신 BFS
연결성 체크(`env_core.verify_connectivity()`와 동일 로직)로 원인을 다시
진단했다.

| 단계 | 도달 불가 셀 | 비고 |
|---|---:|---|
| 원본 추출(개선 전) | 354 / 641 (55%) | |
| + 건물 외곽 마스킹 | 130 / 428 | 외곽 컨투어 밖을 WALL로 채움 |
| + 문 기호 검출 | 8 / 428 | 템플릿 매칭으로 11개 문 발견, 강제로 HALL 처리 |

- **건물 외곽 마스킹**이 최대 원인이었다. 건물 바깥(테두리 밖 여백, 요철
  컷아웃)이 단순 밝기 임계값으로는 그냥 HALL로 남는다. `cv2.findContours`
  (RETR_EXTERNAL)로 건물 폴리곤을 찾아 그 밖을 전부 WALL로 채워 해결했다.
- **문 기호 검출**: 201~209호 각 방과 복도를 잇는 문이 얇은 직사각형
  외곽선으로, 셀 하나보다 작게 그려져 있어 5% 풀링 임계값에서 항상 WALL로
  밀렸다. 문 기호 하나를 템플릿으로 잘라 `cv2.matchTemplate`으로 같은 벽선을
  따라 11개 문을 전부 찾아 해당 셀을 강제로 HALL 처리했다. **이 검출은 이
  도면 특유의 그리기 방식에 맞춘 템플릿 매칭이라, 초록 출구 아이콘(표준
  소방 색상)과 달리 다른 건물 사진에는 그대로 일반화되지 않는다.**
- 남은 8칸은 작은 고립 포켓(화장실 칸막이 등으로 추정)이라 여기서 멈췄다.

## Phase B: 표준 지표 (Zeng et al. 2019 / CubiCasa5K 프로토콜)

| 지표 | 값 |
|---|---:|
| Pixel accuracy | 0.769 |
| WALL IoU | 0.634 |
| WALL precision | 0.702 |
| WALL recall | 0.868 |
| HALL(free-space) IoU | 0.615 |

CubiCasa5K 최신 딥러닝 구현체는 mIoU 87%대를 보고한다(모델·데이터가 달라
직접 비교는 불가하지만, 참고 눈금자로만 인용). 본 프로젝트의 단순 HSV+밝기
임계값 방식은 "전통적 기법" 계열에 속하고, 그 계열이 갖는 한계(정확한 경계
포착 어려움)를 그대로 보인다. WALL recall(0.868)이 precision(0.702)보다
높다는 건, 실제 벽을 놓치는 것보다 벽이 아닌 곳을 벽으로 과탐지하는 쪽이
더 많다는 뜻이다.

## Phase C: 관대한 매칭 (de las Heras et al. 2014 프로토콜, 반경 1셀)

| 지표 | 값 |
|---|---:|
| Lenient recall (r=1) | 0.941 |
| Lenient precision (r=1) | 0.905 |

Phase B의 엄격한 셀 단위 IoU(63%)와 달리, "벽 근처 1칸 이내에 벽이 있으면
인정"하는 관대한 기준으로는 94%/90%까지 오른다. 즉 Phase B에서 드러난 오차
대부분이 완전한 오탐이 아니라 **경계가 1칸 정도 밀린 것**이다. 저해상도
그리드(40×25)에서는 이 정도 경계 오차가 구조적으로 발생하기 쉽다.

## Phase D: 다운스트림 지표 (문헌에 없는 고유 기여)

| 지표 | 값 |
|---|---:|
| 연결성 보존율 | 0.670 |
| EXIT 검출 recall | 1.000 |
| EXIT 검출 precision | 1.000 |

Phase B/C가 "이미지 처리로서 얼마나 정확한가"를 재는 지표라면, 이건 "그
오차가 RL 환경으로 쓰기에 실제로 문제가 되는가"를 잰다. 정답 기준 출구까지
도달 가능한 셀 537개 중 67%가 추출된 그리드에서도 도달 가능했다. 완벽하진
않지만, Phase 0에서 원본 추출의 연결성이 98.1%까지 올라간 것과는 별개
수치다(정답과 예측 사이의 좌표별 일치를 보는 지표라, 두 그리드가 서로 다른
곳에서 약간씩 다르게 뚫려 있으면 개별 연결성은 높아도 이 지표는 낮게 나올
수 있다). **출구 검출은 두 곳 다 정확히 찾아 recall/precision 모두 100%다.**
초록 소방 표지 색상이 표준 색이라 이 결과는 다른 건물 사진에도 일반화될
가능성이 높다(문 기호 검출과 달리).

## 종합

- E2E 파이프라인의 "자동 추출 자체가 동작하는가"는 이번에 실제로 검증됐다.
  출구 검출 100%, 관대한 매칭 기준 90%대, 연결성 45%→98.1%(Phase 0 개선).
- 다만 셀 단위 정밀도(WALL IoU 63%)는 CubiCasa5K급 딥러닝 방법보다 낮다.
  단순 임계값 기반 "전통 기법"의 알려진 한계이며, Liu et al. 2017의
  학습 기반 접근(junction 검출 + 정수계획법, ~90% precision/recall)이
  그 다음 단계로 문헌에 존재한다.
- 문 기호 검출은 이 사진 전용 보정이라 일반화되지 않는다. 다른 건물
  사진에 적용하려면 그 사진의 문 기호를 새로 템플릿화해야 한다.
- 이 평가는 사진 한 장, 사람이 만든 근사 정답 하나에 대한 결과다. 표본이
  1개라 오차의 신뢰구간을 논할 수 없다는 것도 명시적인 한계다.

## 참고 문헌

- S. Zeng, X. Yang, X. Yeung, S. Fu, & W. Chun, "Deep Floor Plan Recognition
  Using a Multi-Task Network with Room-Boundary-Guided Attention," *ICCV*,
  2019.
- A. Kalervo, J. Ylioinas, M. Häikiö, A. Karhu, & J. Kannala, "CubiCasa5K:
  A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis,"
  *arXiv:1904.01920*, 2019.
- L.-P. de las Heras, O. R. Terrades, S. Robles, & G. Sánchez, "Statistical
  segmentation and structural recognition for floor plan interpretation,"
  *IJDAR*, 2014.
- C. Liu, J. Wu, P. Kohli, & Y. Furukawa, "Raster-to-Vector: Revisiting
  Floorplan Transformation," *ICCV*, 2017.
