# Sorting-2와 Stacking ID ①②③⑤ 비교

기학습 EMA 모델과 정규화를 고정하고 태스크별 **100개 ID 배치 × 4회 × 2조건 = 800회**를 비교했습니다. 두 태스크 합계 1,600회입니다. 학습 seed는 42 하나이며 추가 학습·체크포인트 재선택은 없습니다.

| 지표 | Sorting-2 베이스라인 → 조합 | Stacking 베이스라인 → 조합 |
|---|---:|---:|
| 성공률 | 74.00% → 75.00% | 34.75% → 45.25% |
| 성공률 차이와 paired 95% CI | +1.00%p [−3.25, +5.25] | +10.50%p [+3.75, +17.25] |
| 실제 손 XYZ jerk RMS (m/s³) | 41.9177 → 36.3568 (−13.27%) | 11.8274 → 10.8753 (−8.05%) |
| 청크 경계 명령 속도 변화 RMS | 0.17628 → 0.16018 m/s (−9.13%) | 0.37696 → 0.19017 rad/s (−49.55%) |
| 계획 시간 (ms/plan, 실행별 평균의 평균) | 89.85 → 136.78 | 83.61 → 183.93 |

Sorting-2는 성공률 개선이 불확실하며 실제 손 jerk는 감소했습니다. Stacking은 이번 ID 배치에서 성공률과 움직임 품질이 함께 개선됐습니다. 두 태스크의 경계 지표는 XY와 7관절 기준으로 단위가 다릅니다. 계획 시간에는 실행 환경 부하가 포함되며, Stacking은 한 GPU를 네 프로세스가 공유했습니다. 단독 실시간 지연을 측정한 값이 아닙니다.

## 메인 코드 적용 범위

실험 기준 커밋은 `b22441b5ab5d017be82779c259e2787d43ba0d62`입니다. ① 계획 투영, ② PF2MP, ③ 실행 필터, ⑤ LiPo를 켰고 ④ RTC는 껐습니다. 조합 비트는 `16+8+4+1=29`입니다.

- **Sorting-2:** 메인의 `Policy`와 공통 closed-loop 평가 루프를 사용합니다. 태스크/체크포인트 별칭과 블록 이름을 인식하도록 guard를 확장했습니다. 제약은 학습 시연 및 태스크 초기 분포로 보정했습니다. 유효 workspace는 `[0.24, -0.35]`~`[0.80, 0.45]` m, 가속도 14.7 m/s², 물체 envelope margin 0.41 m입니다. 100개 초기 배치의 시작 제약 충족을 확인했으며 전체 경로 가능성을 보장하지는 않습니다.
- **Stacking:** 메인의 XY 구현을 그대로 실행한 것이 아니라 동일 네 방법의 **7관절 공간 확장**입니다. 관측 23차원, 행동 8차원(관절 목표 증분 7개와 예측 그리퍼 폭)을 두 조건 모두 사용합니다. 다음 시점 그리퍼 폭은 학습 정답의 정의이며 미래 관측을 입력으로 읽지 않습니다. 그리퍼는 guidance나 QP로 직접 수정하지 않습니다. 설정의 `feasibility_tolerance_m`은 공통 QP API의 기존 키 이름이며 관절 버전에서는 rad 단위로 해석합니다.

Stacking의 관절 범위·train 99.5분위수×1.1 가속도 한도·PF·LiPo 한도는 기존 관절 버전과 같습니다. 제약 계산과 latent/정규화/QP 반환을 float64로 보존하고, 기준 관절각은 정확한 실행 명령을 사용하도록 수정했습니다. 신경망은 float32입니다. OSQP 수렴 실패 시 동일 최소 수정 QP를 Clarabel로 풀고 같은 **1e-8 rad** 잔차를 검사합니다. 기존 실패 입력 97개 회귀검사를 통과했고, 본 평가의 수치 중단은 0회입니다. Sorting-2의 QP 미수렴 1회는 성공률 분모에 유지했습니다.

## 보고서와 결과 파일

- [공유용 한국어 Word 보고서](sorting2_stacking_id_report_ko.docx)
- [Sorting-2 상세 보고서](sorting2/report_ko.md) · [그림](sorting2/comparison.png) · [summary](sorting2/summary.json)
- [Stacking 상세 보고서](stacking/report_ko.md) · [그림](stacking/comparison.png) · [summary](stacking/summary.json)
- 태스크별 `rollout_metrics.json`은 시계열을 제외한 실행별 집계 지표 800개입니다. `audit.json`과 `archive_verification.json`은 원본 검산·아카이브 해시 대조 기록입니다.
- 원본 시계열, 전체 학습 체크포인트, 머신별 실행 스크립트는 배포하지 않습니다. 기존 공개 [Sorting-2 EMA](../../../fm_baseline/results/sorting2_baseline/README.md)와 [Stacking EMA](../../../fm_baseline/results/stacking_baseline/README.md)는 원본 selected EMA tensor와 바이트 단위로 같습니다. **포장 파일 SHA는 달라질 수 있으며** 관계는 `model_provenance.json`에 있습니다.

## Sorting-2 재실행

원본 시연·공식 분할 목록·D3IL이 필요합니다. 기존 전처리본에 `next_timestamps`가 없으면 갱신한 전처리기로 새 폴더에 다시 생성합니다.

```bash
python preprocess/prepare_sorting.py \
  --raw /path/to/sorting/state --metadata /path/to/sorting/metadata \
  --d3il /path/to/d3il --output data/sorting2_sincos_v1
python -m preprocess.sorting_constraint_view \
  --source data/sorting2_sincos_v1 --output data/sorting_2_sincos_v1
python -m fm_constraint.eval_sorting \
  --config configs/sorting2_combo29.yaml --device cuda:0 \
  --d3il-root /path/to/d3il \
  --contexts-file fm_baseline/results/sorting2_baseline/id_contexts.pkl \
  --start 0 --count 100 --repeats 4 --steps 500 \
  --output fm_constraint/runs/sorting2_combo29
```

공개 baseline의 context 파일은 원래 공식 파일과 같은 SHA인지 `protocol.json`으로 확인합니다. 다른 파일로 대체하면 같은 평가가 아닙니다. 작은 검증은 `--start 140 --count 2 --repeats 1 --steps 20`처럼 본평가와 분리된 smoke 배치를 사용합니다.

고정 수치의 계산 근거는 [constraints_calibration.json](sorting2/constraints_calibration.json)에 있습니다. 학습 파일이 같은지 확인한 뒤 다음 명령으로 재계산할 수 있습니다. reset XY는 smoke 배치에서 측정한 고정 시작점이며 test 성공률로 선택하지 않았습니다.

```bash
python -m fm_constraint.calibrate_sorting \
  --data-dir data/sorting_2_sincos_v1 \
  --reset-xy 0.5249632666754912 -0.2992917961963433 \
  --output fm_constraint/runs/sorting2_calibration
```

## Stacking 재실행

이번 실험에 사용한 [joint_pf_cache.npz](../../../fm_baseline/results/stacking_baseline/joint_pf_cache.npz)를 함께 제공합니다. 아래 명령은 학습 시연을 다시 전처리하거나 캐시를 재생성하지 않고 기존 모델과 공개 캐시로 평가합니다. D3IL 시뮬레이터와 나머지 실행 의존성은 필요합니다.

```bash
python -m fm_constraint.eval_stacking \
  --run-dir fm_baseline/results/stacking_baseline \
  --settings configs/stacking_combo29.json \
  --cache fm_baseline/results/stacking_baseline/joint_pf_cache.npz \
  --d3il-root /path/to/d3il --device cuda:0 \
  --start 0 --count 100 --repeats 4 --steps 1000 \
  --output fm_constraint/runs/stacking_combo29
```

파일은 frozen 실험의 캐시와 바이트 단위로 같으며 SHA256은 `f1c45e9b3f16ff538a9333a9687d96cca83d0c4156fd813367e092f62b7e065c`입니다. [캐시 검증 정보](../../../fm_baseline/results/stacking_baseline/joint_pf_cache.provenance.json)에 배열 규격과 normalizer 대조 결과를 기록했습니다.

캐시를 직접 재생성할 때만 기존 `preprocess/prepare_stacking.py`로 만든 train 1,001개/val 93개 데이터와 공개 train-only normalizer가 필요합니다. `prepare_stacking_joint`는 train에서 가속도 한도와 7D PF support/cache를 계산하며 관절 위치 범위는 D3IL env에서 읽습니다.

```bash
python -m fm_constraint.prepare_stacking_joint \
  --data-dir data/stacking_sincos_v1 \
  --normalizer fm_baseline/results/stacking_baseline/normalizer.npz \
  --d3il-root /path/to/d3il --device cuda:0 \
  --output data/stacking_sincos_v1/pf2mp_cache/joint
```

재생성한 `joint_settings.json`이 공개 설정과 같은지 확인합니다. 다른 데이터나 시뮬레이터 설정으로 재보정한 결과를 이 고정 조건 결과와 혼합하지 않습니다.

`--start`와 `--count`로 나누어도 전역 context ID와 CPU 해시 난수를 유지합니다. 원래 실험은 5개 배치씩 20개 작업으로 나누고 조건 순서를 교차했습니다. 출력은 checkpoint/normalizer/context/settings/cache 서명으로 보호하므로 조건이 다른 실행을 같은 폴더에 혼합할 수 없습니다.

## 지표 재검산과 해석

```bash
python -m fm_constraint.recompute_comparison \
  --report-dir docs/results/sorting2_stacking_combo29_id_20261115/sorting2 \
  --output /tmp/sorting2_paired_stats.json
python -m fm_constraint.recompute_comparison \
  --report-dir docs/results/sorting2_stacking_combo29_id_20261115/stacking \
  --output /tmp/stacking_paired_stats.json
```

이 명령은 공개 실행별 집계에서 성공률·지표 차이·context-paired bootstrap CI를 재계산하며, 원본 시계열에서 jerk/QP를 다시 검산하는 명령은 아닙니다. 원본 검산은 보존된 감사 기록을 구분해 제공합니다. 100개 배치를 10,000회 bootstrap하고 각 배치의 4회 반복을 함께 유지했습니다. 실패 시행은 성공률 분모에서 제외하지 않았습니다.

같은 이미 노출된 ID 배치에서 구현 수정 후 다시 비교한 결과이며 독립된 새 holdout은 아닙니다. 학습 seed 하나의 시뮬레이터 결과로 OOD/실물 성능을 주장하지 않습니다. Stacking 성공 규칙에는 안정 유지 시간이 없습니다. Sorting-2 미러600개 중 공식 직접 대조는101개이며 Stacking 미러1094개의 공식 직접 대조는0개입니다.

보고한 수치는 원격의 원래 frozen 소스에서 나온 결과입니다. 공개 코드는 같은 수치 구현을 모듈로 옮기고 경로·디바이스를 CLI 인자로 바꾼 버전입니다. 공개용 코드의 CPU 회귀검사111개와 별도 smoke를 통과했습니다. smoke는 dev context140 한 개 ×1회 ×5step으로 두 조건을 실행했으며, Sorting-2는CPU, Stacking은GPU를 사용했습니다. 전체1,600회를 공개 포장 후 다시 실행한 결과와는 구분합니다. 상세 내용은 `publication_checks.json`에 있습니다. LiPo 단독 기여는 후속 `①②③ vs ①②③⑤` 비교가 필요합니다.
