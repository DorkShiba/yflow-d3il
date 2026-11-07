# FM Constraint

기존 `fm_baseline` 모델을 고정하고 Projection, PF2MP, VLSA를 독립적으로 선택한다. 설정은 **schema v3**다. PC Correct는 `policy.py`에 `LEGACY_PC` 주석으로 보존했으며 실행하지 않는다. 이전 PC 설정·CLI는 오류로 처리한다.

```text
PF on: 태스크 train KDE 캐시 확인 → 없으면 생성 → 한 번 로드
noise → Predict N회 [FM + 상태 위치 PF → Euler → 행동 Projection]
      → terminal 계획 VLSA → 검사 → 앞 K개 실행 → 재계획
```

PF는 **생성된 미래 상태의 목표 TCP XY**에만 적용한다. `[action(2), obs]`에서 `X[:,1:,2:4]`가 대상이며 action 증분·첫 관측·나머지 상태의 PF 가산항은 항상 0이다. 행동은 이후 FM 평가를 통해 간접적으로 변할 수 있다. Euler가 1회면 이 경로의 행동 변화는 없다. Projection/VLSA는 행동 누적으로 재구성한 목표 위치를 보정한다.

## 구조

| 파일 | 역할 |
|---|---|
| `pf2mp.py` | 전체 train 검증, Gaussian KDE·거리장, 캐시, 위치 전용 gradient |
| `policy.py` | FM+PF → Euler → Projection, terminal VLSA, 실패·비용·채널 검사 |
| `constraints.py`, `qp.py` | 공통 workspace·가속도·물체 영역, float64 OSQP |
| `sample.py`, `evaluation.py`, `eval_*.py` | 공유 캐시 준비 및 오프라인·실환경 평가 |
| `compute_metrics.py`, `render_rollouts.py` | 새/이전 결과 비교와 저장 궤적 재생 |
| `check_constraints.py`, `train.py` | 기존 제약 통계와 baseline 학습 위임 |

## 데이터와 캐시

태스크별 전체 **train** 시연을 하나의 모집단으로 모아 목표 TCP XY 밀도를 추정한다. Pushing과 Inserting을 분리하고 val/test는 fitting에 넣지 않는다. 기본 support는 전체 train 행에서 seed 42로 최대 3,000점을 균등 추출한다.

```text
data/<task>_sincos_v1/pf2mp_cache/<fingerprint>.npz
```

캐시는 Gaussian support, 고밀도 grid 점, 배열 checksum, 데이터 hash와 설정 metadata를 담는다. 추론 시작 때 train 내용·manifest·밀도 설정을 검증하고 없으면 생성한다. Lock과 atomic rename으로 동시 실행을 처리하며 손상 캐시는 격리 후 재생성한다. PF off이면 train과 캐시를 읽지 않는다. Episode나 재계획마다 캐시를 다시 로드하지 않는다.

`bandwidth_m`, support 추출, grid, threshold가 바뀌면 새 캐시를 만든다. `weight`, `distance_alpha`, checkpoint, normalizer만 바뀌면 밀도 캐시를 재사용한다. 첫 생성에서는 전체 train 검증과 threshold 계산을 수행하므로 정상 추론보다 오래 걸린다. 캐시 hit에서도 원본 변경 감지용 hash 순회는 수행한다.

Potential은 `log p(q) + distance_alpha * distance(q, safe_grid_points)`다. 거리장은 연속 level set의 이산 근사이고 Gaussian score는 해석적으로 계산한다. `distance_alpha=0`은 거리항을 뺀 ablation이다. 기본 weight `0.01`, bandwidth `0.025 m`, alpha `-1` 등은 **초기 후보값**이며 성능을 검증한 최적값이 아니다. 밀도는 방문 영역의 주변분포이며 조건부 안전 확률은 아니다.

## 실행

저장소 루트에서 D3IL checkout과 Python 환경을 준비한다. 모델은 `fm_baseline/models/`를 사용하므로 외부 HardFlow는 필요하지 않다. 기본 설정은 CPU다.

```bash
# 기존 baseline과 같은 all-off
python -m fm_constraint.sample --config configs/pushing.yaml --set sample.n_windows=4

# 상태 위치 guidance, 세 방법의 8가지 조합 지원
python -m fm_constraint.sample --config configs/pushing.yaml --pf2mp --projection --vlsa \
  --set sample.n_windows=4
python -m fm_constraint.sample --config configs/inserting.yaml --pf2mp \
  --set methods.pf2mp.weight=0.01 --set sample.n_windows=4

# 실행 8개마다 재계획, 기본 Predict 10회
MUJOCO_GL=disable python -m fm_constraint.eval_pushing --config configs/pushing.yaml --pf2mp --projection
MUJOCO_GL=disable python -m fm_constraint.eval_inserting --config configs/inserting.yaml --pf2mp --vlsa

python -m fm_constraint.check_constraints --config configs/pushing.yaml --calibrate-acceleration
python -m fm_constraint.train --config configs/pushing.yaml
python -m pytest fm_constraint/tests -q
```

CLI > YAML > 기본값 순서다. `--no-pf2mp`, `--no-projection`, `--no-vlsa`로 개별 해제한다. 모든 방법은 학습에서 꺼야 한다. 일반 실험 출력은 `fm_constraint/runs/` 안에만 저장하며 KDE 캐시만 해당 데이터셋의 `pf2mp_cache/`에 저장한다.

## 명시 제약과 중복성

| 제약 | Pushing | Inserting |
|---|---|---|
| Workspace | `[0.21,0.84] × [−0.39,0.49] m` | `[0.01,0.84] × [−0.39,0.74] m` |
| 목표 명령 가속도 | `13.7 m/s²` | `14.2 m/s²` |
| 모든 블록 중심을 포함하는 정8각형 | margin `0.25 m` | margin `0.14 m` |

가속도는 `||a_h-a_(h-1)|| / 0.035²`이며 32면 내접 다각형으로 제한한다. 직전 실제 실행 행동을 다음 계획의 첫 제약에 연결하고 물체 기하는 한 계획 동안 고정한다. 선정 근거는 `results/*_calibration.json`이다.

PF 단독과 all-off는 명시 제약을 **측정만** 한다. Projection 또는 VLSA가 on이면 초기 workspace와 최종 QP 잔차를 강제하며 실패 계획은 실행하지 않는다. 비유한값·PF 계산 실패·solver 실패를 뒤 VLSA가 복구하지 않는다. 명령 제약은 실제 TCP·전체 링크·블록의 물리 안전 증명이 아니다. 접촉 계측이 없으면 `safe_success_rate`는 null이다.

`Projection final + VLSA`는 PF on이어도 같은 연속 투영이므로 거부한다. `Projection every + VLSA`는 조합 비교를 위해 허용하되 마지막까지 같은 제약으로 투영했다면 terminal VLSA는 수치 허용 오차 안에서 중복이다.

`methods.vlsa.execution_recheck=true`이면 실행 직전 최신 관측으로 H=1 QP를 다시 푼다. 수정 행동을 실행한 뒤 남은 큐를 폐기하고 재계획한다. 기본은 false다.

## 결과와 재현

`results.json`, `rollouts.json`, `resolved_config.yaml`에 `pipeline_version`, Predict 내부 순서, terminal 순서, 활성 플래그, 모델·PF·QP 호출/시간, 캐시 식별자를 기록한다. PF 단계의 action 가산항은 0이며 상태 에너지와 실행 행동의 안전을 구분한다. 생성 상태와 행동 누적 목표의 차이도 기록한다. 모델은 PF on/off 모두 기본 10회 호출한다.

```bash
python -m fm_constraint.compute_metrics --eval-dir fm_constraint/runs/<baseline> \
  --eval-dir fm_constraint/runs/<experiment> --label baseline --label pf2mp
python -m fm_constraint.render_rollouts --eval-dir fm_constraint/runs/<evaluation> --gif
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m fm_constraint.render_rollouts \
  --eval-dir fm_constraint/runs/<evaluation> --camera --gif
```

과거 schema v2 결과는 기존 순서 metadata로 읽으며 PF 결과로 재해석하지 않는다. 2026-10-29 PC 통합 검증은 `results/order_projection_pc_vlsa_validation.json`, 이전 이관 결과는 `results/legacy/`에 보존한다. PF2MP 설계·구현 진행은 [PLAN 8절](../docs/PLAN.md), 논문 요약은 [PF2MP.md](../docs/PF2MP.md)를 따른다.

## ①·②·③의 16개 순차 실험

[scripts/run_method123.sh](../scripts/run_method123.sh)는 Pushing 8개 → Inserting 8개를 순차 실행한다. ④ RTC·⑤ LiPo와 terminal VLSA는 OFF다. ③은 별도 `execution_filter` 옵션으로 **실행 직전 H=1 QP만** 사용한다. 필터 수정이 허용 오차를 넘으면 해당 행동 실행 후 남은 큐를 폐기한다.

```bash
conda activate yflow

# 실행하지 않고 공통 context·seed 계획과 16개 명령만 확인
bash scripts/run_method123.sh --dry-run

# 동작 확인: 각 5 context × seed 100,101 × 8조합, 총 160 rollout
bash scripts/run_method123.sh --stage smoke

# 예비 비교: 각 10 tuning context × seed 100,101,102, 총 480 rollout
bash scripts/run_method123.sh --stage pilot

# 본실험: 각 60 context × seed 0..4 × 8조합, 총 4,800 rollout
bash scripts/run_method123.sh --stage main --output fm_constraint/runs/method123_main

# 전체 Inserting val 82개를 사용할 경우: 총 5,680 rollout, validation 평가로 표시
bash scripts/run_method123.sh --stage main --inserting-set all-val \
  --output fm_constraint/runs/method123_all_val
```

활성 환경 대신 interpreter를 지정하려면 `PYTHON=/path/to/python`을 앞에 붙인다. `--device cuda:0`도 지원하지만 이 스크립트의 기능 검증은 CPU에서 수행했다. `MUJOCO_GL` 기본값은 `disable`이며, D3IL은 외부 checkout의 모델 폴더에 임시 XML을 쓸 수 있어야 한다.

기본 Inserting 분할은 val의 80개 context group을 seed 2026으로 20개 조정용/60개 최종 평가용으로 나누고 그룹별 대표 1개를 선택한다. Pushing은 조정 시 train contexts, 본실험 시 test 60개를 사용한다. 이미 전체 val로 파라미터를 조정했다면 이 분할만으로 독립 최종 평가가 되지는 않는다. 목록은 `*_tuning_contexts.json`, `*_final_contexts.json`, `*_selected_contexts.json`에 저장한다.

조합 ID는 보고서의 `P00 P04 P08 P12 P16 P20 P24 P28`, 이어서 같은 번호의 `I`다. 같은 `(task, context_id, sampling_seed)`에는 같은 FM RNG를 사용하며 조합 ID·이전 rollout 길이·캐시 생성과 독립적이다. 재계획 횟수가 달라지면 같은 계획 인덱스의 noise만 대응하며 이후 관측이 같다는 뜻은 아니다.

출력에는 동결한 설정·코드/데이터/모델 hash, 조합별 console 로그, `results.json`, `rollouts.json`, 초기 조건 snapshot, 완료 표식이 포함된다. 마지막에 `summary.md`/`summary.json`으로 성공률·실행 step 위반율·행동 변화·명령 jerk·조건별 entropy와 2,000회 context bootstrap CI·대응 차이·Δ를 집계한다. Δ 분해는 PF+Filter, Projection+Filter, Projection+PF 및 `(Projection+PF)+Filter`로 사전 고정한다. 실제 접촉 계측이 없으면 안전 성공률은 null이다.

동일 옵션과 `--output`으로 재실행하면 검증된 완료 조합은 건너뛴다. 코드·입력·의존 패키지가 달라지면 결과 혼합을 막기 위해 중단한다. 프로세스 실패 시 후속 조합을 중단하고, 재개 시 부분 출력은 `*.interrupted.*`로 보존한다. 방법 자체의 계획/필터 실패는 성공률 분모에 포함하여 완료 결과로 저장한다. 기본 스크립트는 본실험을 실제로 시작하므로 명령 확인에는 `--dry-run`을 사용한다.


## 내부 모델 이관 (2026-11-07)

모델·CFM·조건부 Euler는 `fm_baseline/models/`로 이관했다. 외부 모델 경로·설치 없이 기존 EMA checkpoint를 읽는다. 구 YAML의 `paths.hardflow_root`는 읽을 때 제외하며 기존 checkpoint config의 같은 이름도 실행에 사용하지 않는다. 출처는 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), 수치 비교 근거는 [검증 요약](results/local_models_validation.json)에 기록한다.

새 순차 실험 manifest는 내부 모델 파일과 baseline 로더·trainer의 hash도 동결한다. **이관 전 manifest의 코드 hash를 무시하거나 갱신하지 않는다.** 기존 결과는 보존되며 집계·재생할 수 있다. 이전 동결 실험의 실행 재개는 원래 코드 snapshot에서 수행하고, 이관된 코드의 새 실행은 새 `--output` 경로를 사용한다.

## Inserting을 GPU 두 개에 4조합씩 분할

새 스크립트 [run_inserting_gpu_group.sh](../scripts/run_inserting_gpu_group.sh)는 기존 runner와 manifest를 그대로 사용한다. 기존 8개 순차 실행을 먼저 중단한 뒤, GPU 하나씩 할당받은 두 셸에서 실행한다. 두 작업이 같은 파일시스템의 동일 출력 경로를 사용해야 완료 결과와 잠금을 공유할 수 있다.

```bash
# GPU 할당 셸 1: conda activate yflow 후 실행
bash scripts/run_inserting_gpu_group.sh --group 1 \
  --output fm_constraint/runs/inserting_gpu_main

# GPU 할당 셸 2: conda activate yflow 후 실행
bash scripts/run_inserting_gpu_group.sh --group 2 \
  --output fm_constraint/runs/inserting_gpu_main
```

| 그룹 | 순차 실행 조합 | 기본 계획량 |
|---|---|---:|
| 1 | I00 → I04 → I08 → I12 | 60 context × 5 seed × 4 = 1,200회 |
| 2 | I16 → I20 → I24 → I28 | 60 context × 5 seed × 4 = 1,200회 |

Slurm이 각 셸에 GPU 하나를 노출하면 두 스크립트 모두 기본 `cuda:0`을 사용한다. 같은 manifest에서 그룹 2만 `--device cuda:1`로 바꾸지 않는다. 한 셸에 GPU 두 개를 모두 할당받은 경우에만 다음처럼 프로세스별 노출 장치를 나눈다.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_inserting_gpu_group.sh --group 1 \
  --output fm_constraint/runs/inserting_gpu_main > inserting_group1.log 2>&1 &
group1_pid=$!
CUDA_VISIBLE_DEVICES=1 bash scripts/run_inserting_gpu_group.sh --group 2 \
  --output fm_constraint/runs/inserting_gpu_main > inserting_group2.log 2>&1 &
group2_pid=$!
wait "$group1_pid"
wait "$group2_pid"
```

Slurm이 이미 장치를 제한한 별도 할당 셸에서는 `CUDA_VISIBLE_DEVICES`를 임의로 덮어쓰지 않고 위의 그룹별 명령을 사용한다. GPU 모델·호스트·Slurm 작업 ID·노출 장치는 `group_1.device.log`, `group_2.device.log`에 기록한다.

- `--output`은 이전 GPU 8조합 실행의 경로와 같게 지정한다. 기본값은 `fm_constraint/runs/inserting_gpu_main`이다. 이전 실행이 `all-val` 또는 다른 split seed를 사용했다면 같은 옵션을 지정한다.
- 완료 표식과 결과 checksum이 맞는 조합은 건너뛴다. 중단된 조합은 `*.interrupted.*`로 보존하고 300회 전체를 재실행한다. Rollout 중간부터 재개하지 않는다.
- Shared suite lock은 두 그룹의 병렬 실행을 허용하고 기존 전체-suite 스크립트의 실행과 충돌한다. 그룹별 exclusive lock은 같은 그룹의 이중 실행을 막는다. 기존에 직접 작성한 `for` 루프나 직접 Python 실행은 이 잠금을 사용하지 않으므로 반드시 먼저 중단한다.
- 최초 manifest 생성은 별도 lock으로 직렬화한다. 두 그룹은 공통 context·seed·설정과 기존 캐시 lock을 공유한다. 기존 manifest의 코드 hash를 수정하거나 검증을 우회하지 않는다.
- `--dry-run`이면 GPU와 시뮬레이터를 실행하지 않고 해당 그룹의 4개 명령을 출력한다. `--stage smoke --context-limit 1 --max-steps 2` 등 검증 옵션도 지원한다.
- 이 스크립트는 Inserting 8개만 실행하므로, 전체 16개 완료가 필요한 기존 `summarize`는 자동 호출하지 않는다. 각 조합의 결과는 원래 형식으로 저장된다.
