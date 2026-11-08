# 재엽 담당 ID 실험 결과

2026년 11월 8일 완료한 context 17–19의 Pushing 및 Inserting 실험이다. 태스크별 32조합과 noise seed 0·1·2를 평가해 **192개 단위, 576개 rollout**을 저장했다. 각 조합은 3 context × 3 seed = 9회이며, 수치는 담당 범위의 점추정이다.

## 보고서

- [최종 보고서 Word 문서](final_report/YFlow_재엽_ID실험_최종보고서_20261108.docx): 실험 설계, 주요 결과, PF2MP·LiPo 병용 분석, 궤적 그림, 한계, 전체 64조합 표를 포함한 9쪽 보고서
- [팀원별 자동 집계 보고서](../../docs/report_jaeyeop.md)
- [재집계 수치](final_report/report_metrics.json)
- [실험 당시 프로토콜](final_report/protocol_4d33329.md)

## 결과와 분석 자료

| 경로 | 내용 |
|---|---|
| [`experiments/id_v1/`](../../experiments/id_v1/) | 팀 통합 집계용 단위별 `result.json`과 `trajectories.npz` |
| [`id_v1/`](id_v1/) | 이번 보고서에 사용한 재엽 담당 결과의 고정 사본 |
| [`pf2mp_lipo_analysis/`](pf2mp_lipo_analysis/) | 72개 rollout 비교 수치, 궤적, potential 진단, 분석 스크립트 |
| [`full_logs/`](full_logs/) | 본실험 전체 계획 로그와 평가 부속 파일을 보존한 6개 압축 파일 |
| [`runtime/`](runtime/) | 실행·종료 기록, 환경 패키지 목록, 테스트 및 동작 확인 로그 |
| [`final_report/result_manifest.json`](final_report/result_manifest.json) | 보고서의 원본 단위별 결과 192개의 SHA256 |

`full_logs/manifest.json`에는 압축 파일과 내부 파일별 크기 및 SHA256이 기록돼 있다. 각 압축 파일은 태스크와 context 하나의 32조합을 담고 있으며, 압축 내부 경로는 `<task>/<combo>/ctx_<index>/...`이다. 별도 디렉터리에 다음과 같이 풀 수 있다.

```bash
mkdir -p /tmp/jaeyeop-full-logs
for archive in outputs/jaeyeop_20261108/full_logs/*.tar.gz; do
  tar -xzf "$archive" -C /tmp/jaeyeop-full-logs
done
```

## 결과 해석

Pushing의 RTC + LiPo는 9/9 성공했고 baseline 대비 명령 jerk가 약 30% 낮았다. Inserting의 Projection + PF2MP는 9/9 성공했으며, PF2MP + RTC는 8/9 성공과 측정 제약 위반율 0%를 보였다. PF2MP + LiPo는 Inserting에서 단독 방법 각각의 8/9 성공에 비해 2/9로 낮았다. 전체 ON은 두 태스크 모두 7/9 성공했다.

병용 실패 로그는 작은 행동 보정 이후 접촉·관측·재계획 과정에서 경로가 달라지는 설명과 부합한다. 정확한 접촉 수준의 원인은 확정하지 않았다. 3 context 결과만으로 통계적 유의성이나 전체 환경의 우열을 주장하지 않는다. Inserting은 validation 초기 상태 평가다.

## 재현 정보

- 실험 코드: `4d33329fe332a6c7687d5bb9e25d36997fb61344`
- 코드 fingerprint: `0167b01b63c192f42f6ca1af9813d6d64b923cab17ed6642d4e8c5faffabb669`
- D3IL: `1d9c71850d4fc1477cee17b557c8b97d70b13071`
- 장비: RTX 3090 24GB 한 장, context별 세 프로세스
- 본실험: 2026년 11월 8일 13:41:04–16:57:10 KST
- 방법 실패: Inserting I16·I18·I20·I22의 context 17·seed 0에서 solver 실패 각 1회. 실패를 분모에 포함했다.

스크립트에 기록된 `/root/d3il-work/`와 로컬 경로는 실행 당시 배치다. 다른 환경에서 분석·보고서를 다시 생성하려면 경로를 맞추고 필요한 Python 패키지를 설치해야 한다. `runtime/`의 24-step 동작 확인은 본실험 576회 결과에 포함하지 않았다.
