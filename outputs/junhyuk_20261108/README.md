# 준혁 담당 ID 실험 결과

2026년 11월 08일 완료한 context 7–16의 Pushing 및 Inserting 실험이다. 태스크별 32조합과 noise seed 0·1·2를 평가해 **640개 단위, 1920개 rollout**을 저장했다. 각 조합은 10 context × 3 seed = 30회이며, 수치는 담당 범위의 점추정이다.

## 보고서

- [팀원별 자동 집계 보고서](../../docs/report_junhyuk.md) (사본: [`report_junhyuk.md`](report_junhyuk.md))
- [실험 당시 프로토콜](final_report/protocol_4d33329.md)

## 결과와 분석 자료

| 경로 | 내용 |
|---|---|
| [`experiments/id_v1/`](../../experiments/id_v1/) | 팀 통합 집계용 단위별 `result.json`과 `trajectories.npz` |
| [`id_v1/`](id_v1/) | 준혁 담당 결과의 고정 사본 |
| [`runtime/`](runtime/) | 실행 스크립트, 프로세스별 로그, 시작·종료 시각, 환경 패키지 목록, GPU 정보, 설치 확인·스모크 로그 |
| [`final_report/result_manifest.json`](final_report/result_manifest.json) | 원본 단위별 결과 640개의 SHA256 |

## 요약 수치

| 태스크 | baseline(00) 성공 | 전체 ON(31) 성공 |
|---|---:|---:|
| Pushing | P00 29/30 | P31 26/30 |
| Inserting | I00 27/30 | I31 21/30 |

### 방법별 효과 (나머지 4개 설정이 같은 16쌍의 켬 − 끔 평균)

| 방법 | Pushing Δ성공 %p | Pushing Δjerk | Inserting Δ성공 %p | Inserting Δjerk |
|---|---:|---:|---:|---:|
| ① Projection | -0.8 (상승 7/16) | -7% | -6.0 (상승 4/16) | -10% |
| ② PF2MP | -1.2 (상승 5/16) | +1% | +0.2 (상승 7/16) | -1% |
| ③ Filter | -0.4 (상승 2/16) | -0% | -1.9 (상승 4/16) | -0% |
| ④ RTC | -11.2 (상승 0/16) | -17% | -15.6 (상승 0/16) | -23% |
| ⑤ LiPo | +2.1 (상승 6/16) | -17% | -3.1 (상승 6/16) | -18% |

## 결과 해석

- **④ RTC**는 jerk를 가장 크게 줄이지만 성공률을 일관되게 낮췄다. 두 태스크의 32쌍 중 RTC를 켜서 성공률이 오른 쌍이 없다. 이전 청크에 강하게 맞추는 설정(β=5)이 계획 수정을 막는 것으로 추정한다.
- **⑤ LiPo**는 성공률 변화가 작은 상태에서 jerk를 줄인 방법이다.
- **④+⑤**는 태스크마다 반대로 나타났다. Pushing에서는 LiPo가 RTC의 성공률 하락을 일부 회복하면서 jerk가 가장 낮았고, Inserting에서는 병용 시 성공률이 RTC 단독보다 더 낮았다.
- **③ Filter**는 ID 조건에서 거의 개입하지 않아, 켬·끔 결과가 소수점까지 같은 쌍이 있다.
- **② PF2MP**는 성공률·jerk 효과가 작았다. Pushing에서 ① 없이 켜면 제약 위반이 생겼고 Inserting에서는 엔트로피가 낮아졌다.
- **① Projection**은 Inserting에서 성공률을 낮췄고, 방법 실패(solver 실패)는 모두 ①이 켜진 Inserting 조합에서 나왔다.
- jerk가 가장 낮은 조합은 대부분 ④를 포함해 성공률이 낮다. 성공률을 유지한 절충은 Pushing P03(④+⑤), Inserting I25·I29(①+②+⑤, ③ 유무)였다.

전체 64조합 수치는 자동 집계 보고서에 있다. 담당 10개 context × 3 seed(조합당 30회)의 점추정이며, 1회 차이가 3.3%p다. 통계적 유의성과 조합 간 상보성은 팀 전체 30 context 통합 결과로 판단한다.

## 재현 정보

- 실험 코드: `4d33329fe332a6c7687d5bb9e25d36997fb61344`
- 코드 fingerprint: `0167b01b63c192f42f6ca1af9813d6d64b923cab17ed6642d4e8c5faffabb669`
- D3IL: `1d9c71850d4fc1477cee17b557c8b97d70b13071`
- 장비: Tesla V100-DGXS-32GB (cuda:0, cuda:1, cuda:2, cuda:3), 단위 잠금으로 프로세스 여러 개가 같은 범위를 나눠 실행
- 본실험: 2026년 11월 08일 14:01:53–2026년 11월 08일 23:22:56 KST
- 방법 실패(분모에 포함):
  - Inserting solver_failure 7회: I19 ctx 14 seed 2, I23 ctx 10 seed 2, I24 ctx 14 seed 2, I25 ctx 16 seed 2, I26 ctx 14 seed 2, I28 ctx 14 seed 2, I29 ctx 16 seed 2

스크립트에 기록된 `/workspace/`와 로컬 경로는 실행 당시 컨테이너 배치다. `runtime/`의 24-step 동작 확인은 본실험 결과에 포함하지 않았다.
