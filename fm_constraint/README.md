# FM Constraint: train-free 5개 방법

기존 `fm_baseline` 모델을 고정하고 다섯 방법을 독립적으로 켜고 끈다. 설정은 **schema v4**다.

| # | 방법 | 위치 | 키 | 조합 비트 |
|---|---|---|---|---:|
| ① | Projection | Predict 매 Euler 직후(`mode: every`) 또는 끝 1회(`final`), 행동 공동 투영 QP | `methods.projection` | 16 |
| ② | PF2MP | Predict 벡터장에 상태 위치 potential guidance (행동 가산항 0) | `methods.pf2mp` | 8 |
| ③ | Filter | 실행 직전 최신 관측으로 행동 하나를 H=1 QP. 수정하면 남은 큐를 버리고 재계획 | `methods.filter` | 4 |
| ④ | RTC | Predict 벡터장에 직전 계획의 미실행 행동 suffix inpainting guidance (`beta`) | `methods.rtc` | 2 |
| ⑤ | LiPo | 계획 끝 고정 한도(`bound_m`, 기본 0.3 mm) minimum-jerk QP. ①이 켜져 있으면 ① 제약을 같은 QP에 포함 | `methods.lipo` | 1 |

```text
noise → Predict N회 [FM(+④) + ② → Euler → ①] → ⑤ → 큐 → 행동마다 [③] → env.step
                                                    ↑ 앞 K=8개 실행 후 재계획
```

- 실행 순서는 고정이며 결과에 `pipeline_order: [rtc, pf2mp, projection, lipo, filter]`로 기록한다.
- 이전 terminal-plan VLSA는 ① `mode: final`과 같은 연산이라 삭제했다(`docs/PLAN.md` 10절). 원래 의미의 VLSA는 ③이다.
- ④의 참조: 활성 계획에서 아직 실행하지 않은 H−s개 행동. 첫 계획, 그리고 ③이 행동을 수정해 큐를 버린 직후에는 참조가 없어 ④가 적용되지 않는다(`rtc_guided`로 기록).
- 추론 지연(OOD 조건 E, d step)이 있으면 다음 계획을 d step 일찍 세우고 새 chunk의 index d부터 실행한다. 그 사이 실행될 d개 행동은 ④ mask에서 1(고정), ⑤에서 등식으로 고정한다.
- ⑤는 interior-point(Clarabel)로 mm 단위에서 푼다. ①·③의 투영 QP는 OSQP다. ⑤에 ① 제약을 넣을 때, 보정 한도로는 닿을 수 없는 면(여유 > ‖G_row‖₁·2·bound)은 해를 바꾸지 않으므로 빼고 푼다. 최종 판정은 전체 면으로 한다.

## 실행

```bash
# 오프라인 계획 (all-off = baseline과 동일)
python -m fm_constraint.sample --config configs/pushing.yaml --projection --lipo --set sample.n_windows=4

# D3IL 평가: 원하는 방법만 켠다. 8 step마다 재계획, Predict 10회
MUJOCO_GL=disable python -m fm_constraint.eval_pushing --config configs/pushing.yaml --projection --filter --rtc
MUJOCO_GL=disable python -m fm_constraint.eval_inserting --config configs/inserting.yaml --pf2mp --lipo

# OOD 조건 (ood/README.md): context·로봇 시작·노이즈·push·지연·정책 seed를 data/ood에서 읽는다
MUJOCO_GL=disable python -m fm_constraint.eval_pushing --config configs/pushing.yaml --ood E_mid --rtc --lipo

python -m pytest fm_constraint/tests tests -q
```

CLI > YAML > 기본값 순서다. `--no-<method>`로 개별 해제한다. 학습에서는 모든 방법을 꺼야 한다. 출력은 `fm_constraint/runs/` 안에만 저장한다.

## 32조합 순차 실험

```bash
bash scripts/run_suite.sh --dry-run                     # 공통 context·seed 계획과 64개 명령만 확인
bash scripts/run_suite.sh --stage smoke                 # 태스크별 5 context × seed 2개 × 32조합
bash scripts/run_suite.sh --stage main --output fm_constraint/runs/suite_main
```

조합 ID는 `P00`~`P31`, `I00`~`I31`이다(비트 ①16 ②8 ③4 ④2 ⑤1). 이전 method123 실험의 `P00 P04 … P28`은 ④⑤ OFF인 같은 번호이며, 2026-11-07 기준 코드에서 이 8조합의 실행 행동은 이전 코드와 비트 단위로 같다. Complementarity Δ의 비교 쌍은 실행 전에 고정했다: 조합에서 가장 낮은 비트의 방법을 B, 나머지를 A로 둔다(예: 28 = (24, 4), 31 = (30, 1)).

같은 `--output`으로 재실행하면 검증된 완료 조합은 건너뛰고, 코드·입력·의존 패키지가 바뀌었으면 중단한다. 실험 조건 통일(OOD 조건을 runner에 넣는 방식 포함)은 브랜치 통합 이후에 정한다.

## 명시 제약

| 제약 | Pushing | Inserting |
|---|---|---|
| Workspace | `[0.21,0.84] × [−0.39,0.49] m` | `[0.01,0.84] × [−0.39,0.74] m` |
| 목표 명령 가속도 | `13.7 m/s²` | `14.2 m/s²` |
| 모든 블록 중심을 포함하는 정8각형 | margin `0.25 m` | margin `0.14 m` |

가속도는 `||a_h−a_(h−1)|| / 0.035²`이며 32면 내접 다각형으로 제한한다. 직전 실제 실행 행동을 다음 계획의 첫 제약에 연결하고 물체 기하는 한 계획 동안 고정한다. 선정 근거는 `results/*_calibration.json`이다. ①이 꺼져 있으면 계획 단계에서는 제약을 측정만 한다. 명령 제약은 실제 TCP·링크·블록의 물리 안전 증명이 아니다.

## PF2MP 캐시

태스크별 전체 **train** 시연의 목표 TCP XY로 Gaussian KDE와 고밀도 영역 거리장을 만들고 `data/<task>_sincos_v1/pf2mp_cache/<fingerprint>.npz`에 캐시한다(lock·atomic rename). `bandwidth_m`·support·grid·threshold가 바뀌면 새로 만들고 `weight`·`distance_alpha`만 바뀌면 재사용한다. 기본값(weight 0.01, bandwidth 0.025 m, alpha −1)은 초기 후보값이다.

## 파일

| 파일 | 역할 |
|---|---|
| `policy.py` | Predict[FM(+④)+② → Euler → ①] → ⑤, 실패·비용·채널 검사 |
| `rtc.py`, `lipo.py`, `pf2mp.py` | ④ soft mask·guidance, ⑤ jerk QP, ② KDE·거리장·캐시 |
| `constraints.py`, `qp.py` | 공통 workspace·가속도·물체 영역 행렬, OSQP 투영 |
| `evaluation.py`, `eval_*.py`, `sample.py` | D3IL 평가 루프(③, ④ 참조, 지연, OOD), 오프라인 계획 |
| `suite.py`, `suite_metrics.py` | 32조합 순차 실험과 context bootstrap 집계 |
| `tasks.py` | D3IL 8개 태스크 정의(OOD 기하, 추가 태스크 baseline 평가) |
| `compute_metrics.py`, `render_rollouts.py`, `check_constraints.py`, `train.py` | 결과 비교·재생, 제약 통계, baseline 학습 위임 |

결과 디렉터리에는 `results.json`, `rollouts.json`, `resolved_config.yaml`이 남는다. 계획마다 모델·PF·QP 호출과 시간, `rtc_guided`, `frozen_prefix`, ⑤ 보정량·jerk 비용, 실행 step마다 `seam`·③ 개입·실제 상태 기준 위반량을 기록한다.
