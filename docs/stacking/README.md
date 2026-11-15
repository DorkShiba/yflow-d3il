# Stacking 본실험: 32조합 × ID·OOD A

Stacking에서 ①②③④⑤ 32조합을 ID와 A_weak/A_mid/A_strong에서 비교한다(2026-11-15 결정). Pushing·Inserting 본실험과 코드 fingerprint·결과·보고서를 분리했다.

| 항목 | 내용 |
|---|---|
| 모델 | `fm_baseline/results/stacking_baseline/baseline_ema.pt` (35만 step EMA, seed 42), train-only normalizer. 재학습·재선택 없음 |
| 방법 | `fm_constraint/joint_combo.py`: 팀원 관절 공간 ①②③⑤(`joint.py`)를 방법별로 켜고 끄도록 일반화하고, ④ RTC를 추가함. 수치는 `configs/stacking_combo29.json` |
| ④ RTC | main `rtc.guided_velocity`. 직전 계획의 미실행 8개 행동을 target으로 둔다. **그리퍼 채널 가중치 0**. 지연 0, β 5 |
| 그리퍼 | 어떤 guidance·QP도 직접 수정하지 않는다 |
| context | `id_contexts.pkl` test [0,100)에서 고정 seed로 30개 (`configs/stacking_experiment_contexts.json`) |
| 조건 | `id`, `A_weak` 1 cm, `A_mid` 5 cm, `A_strong` 18 cm (옮길 수 있는 블록만 옮김), 데이터는 `data/ood_stacking` |
| rollout | 단위 = (조건, 조합, context). noise seed 0·1·2로 rollout 3개, 최대 1,000 step, plan 16 / execute 8 / Euler 10, dt 0.03 s |
| FM noise | 팀원 비교 실험과 같은 `sha256(["stacking","test_NNN",seed])` 규칙. OOD도 ID와 같은 noise |

## 실행

팀원 분담과 기본 명령은 루트 [README.md](../../README.md) 11절 "Stacking 분담"에 있다. 보통은 `bash scripts/run_stacking_experiment.sh --name <이름> --contexts <범위>`로 실행한다. 이 스크립트는 코드·GPU 검사, 로그 저장, 조건별 보고서 생성까지 한다. 아래는 스크립트가 부르는 Python 명령이다.

```bash
# 동작 확인 (결과는 fm_constraint/runs/experiment_stacking_v1/smoke에만 저장)
MUJOCO_GL=disable python -m fm_constraint.stacking_experiment run --cond id --combos 0,29 \
    --contexts 0 --device cuda:0 --name hyeongbin --max-steps 40

# 본실험: 맡은 context × 원하는 조건을 순서대로 실행. 중단 후 같은 명령이면 완료된 단위를 건너뛴다
MUJOCO_GL=disable python -m fm_constraint.stacking_experiment run --cond id,A_weak,A_mid,A_strong \
    --combos all --contexts 0-7 --device cuda:0 --name hyeongbin

# 보고서 (git에는 이것만 올린다): docs/stacking/report_<cond>_<name>.md
python -m fm_constraint.stacking_experiment report --cond A_weak --name hyeongbin
# 통합 (누구나, 보고서만 읽음): docs/stacking/report_<cond>_results.{md,json}
python -m fm_constraint.stacking_experiment merge --cond A_weak
```

- 같은 장비에서 여러 프로세스를 띄워도 단위마다 잠금을 잡는다. 시뮬레이터가 CPU 시간 대부분을 쓰므로 GPU 하나에 2–4개를 띄워도 된다. 결과 파일이 있으면 그 단위를 건너뛴다.
- 결과에 영향을 주는 파일(`CODE_FILES`)에 커밋하지 않은 변경이 있으면 본실험을 시작하지 않는다.
- PF 캐시: 팀원이 공개한 `fm_baseline/results/stacking_baseline/joint_pf_cache.npz`(2026-11-15 조합 29 비교에 쓴 원본, sha256 `f1c45e9b…`)를 `configs/stacking_experiment.json`에 고정했다. 다른 파일이면 runner가 본실험을 거부한다. `--pf-cache`는 동작 확인용이다.

## 지표

| 지표 | 정의 |
|---|---|
| 성공 % | 세 블록이 목표 XY 6 cm 안, 모든 쌍의 Z 차이 > 3 cm (D3IL 판정, 유지 시간 없음) |
| 목표 진입 1개 / 2개 % | D3IL `success_1/2`. 서로 다른 블록이 목표 영역에 들어간 **이력**이며, 안정된 1단·2단 쌓기가 아니다 |
| 위반 % | 실제 실행한 관절 증분이 실행 직전 목표 관절과 직전 실행 증분 기준의 **관절 위치 범위(D3IL robot)·축별 가속도 한도**(H=1)를 1e-8 rad 넘게 벗어난 step 수 / 실행 step 수. 유형별 수는 `result.json`에 있다 |
| action change mrad, jerk rad/s³ | main과 같은 형태로, 연속 관절 행동 차이 norm의 평균, 2차 차분 norm의 평균을 dt³로 나눈 값 |
| 손 jerk m/s³, 경계 변화 rad/s | 실제 TCP 3차 차분 RMS, 재계획 경계의 관절 명령 속도 변화 RMS (팀원 정의) |
| entropy | 성공한 rollout의 쌓은 순서(6가지) 정규화 entropy |
| 방법 실패 % | 투영·필터·LiPo 실패로 중단된 rollout. 분모에 남긴다 |

merge는 30 context bootstrap(2,000회) 95% CI, Complementarity Δ(main과 같은 쌍 규칙), A 조건의 **옮겨진 블록 수별 성공률**을 낸다. 이동 방향(x±/y±)별 분석은 보고서 원시 데이터의 `faces`를 쓴다.

## 선택된 context와 A_strong

선택된 30개: test 1, 2, 3, 6, 19, 23, 31, 34, 35, 37, 41, 42, 45, 50, 55, 57, 59, 62, 69, 70, 72, 75, 77, 82, 87, 92, 93, 94, 97, 98.

A_strong에서 옮겨진 블록 수는 1개 18 context, 2개 8 context, 3개 4 context다. 빨강은 항상 y−로 18 cm 옮겨진다. 초록(목표 영역·빨강 범위에 막힘)과 파랑(팔 도달 범위 밖)은 일부만 옮겨진다. 블록별 최대 가능 거리는 `data/ood_stacking/max_shift.json`에 있다.

## 검증 (2026-11-16)

- 같은 기기에서 조합 0과 29의 1,000 step rollout(test_000, seed 0)이 팀원 `eval_stacking.py`와 행동·관절 궤적까지 비트 단위로 같다(합성 PF 캐시 사용).
- 단위 테스트(`fm_constraint/tests/test_stacking_experiment.py`): 조합 0·4·29와 팀원 `JointPolicy`의 동등성, ④가 그리퍼 target을 무시하는지, ① 없는 ⑤, 집계식.
- 공개 캐시로 조합 29를 돌린 test_000·test_001 seed 0은 공개 결과와 성공·목표 진입·step 수(756, 1000)가 같다.
- 공개 수치(3090)와 로컬(3050 Ti)은 같은 seed에서 성공·step 수는 같지만 jerk가 조금 다르다. GPU float32 연산 차이로 보인다. 그래서 **한 context의 모든 조합은 같은 장비에서 돌리도록** context 단위로 분담한다.

## 비용

로컬 3050 Ti에서 계획 1회는 조합에 따라 0.2–0.6 s였다(팀원 3090에서 조합 29는 0.18 s). 시뮬레이터 1 step은 12 ms다. 전체는 3,840 단위(11,520 rollout)로, 3090 기준 약 80 GPU 시간으로 추정한다.
