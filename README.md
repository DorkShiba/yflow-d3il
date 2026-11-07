# Y-Flow D3IL: 플로우 매칭 베이스라인과 제약 실험

연세대학교 인공지능종합설계 Y-Flow 팀의 D3IL 실험 저장소다. 상태 기반 플로우 매칭 정책을 D3IL **Pushing**과 **Inserting** 태스크로 학습하고, 물리 제약 주입 방식들을 조합해 비교하는 것이 목표다.

순수 플로우 매칭 베이스라인 위에 추론 시점에만 적용하는(train-free) 5개 방법 **① Projection, ② PF2MP, ③ 실행 직전 필터, ④ RTC, ⑤ LiPo**를 독립적으로 켜고 끌 수 있다. OOD 조건 20개에서도 평가할 수 있다. 모델·CFM·Euler 샘플러는 `fm_baseline/models/`에 있으며 외부 HardFlow는 필요하지 않다.

| 문서 | 내용 |
|---|---|
| [`docs/METHODS.md`](docs/METHODS.md) | 5개 방법과 제약의 현재 정의, 실행 순서, 상호작용 규칙 |
| [`fm_constraint/README.md`](fm_constraint/README.md) | 5개 방법 실행·설정, 32조합 runner |
| [`ood/README.md`](ood/README.md) | OOD 조건(A 블록 위치, B 로봇 시작, C 관측 노이즈, D push, E 추론 지연)과 seed 규칙 |
| [`fm_baseline/README.md`](fm_baseline/README.md) | 베이스라인 학습·평가 기록, 실패 분석 |
| [`data/README.md`](data/README.md), [`docs/README.md`](docs/README.md) | 데이터 출처·전처리·분할, 관측·행동 정의 |
| [`docs/PLAN.md`](docs/PLAN.md) | 브랜치 통합 계획과 결정 기록 |

## 1. 저장소 구조

```text
yflow-d3il/
├── data/                       # 전처리된 상태·행동 데이터 (시연별 npz)
│   ├── pushing_sincos_v1/      # train 1,800 / val 200
│   ├── inserting_sincos_v1/    # train 718 / val 82
│   └── ood/                    # OOD 평가 사례 (pushing/inserting.jsonl, manifest.json)
├── fm_baseline/
│   ├── models/                 # 내부 TemporalUnet·CFM·조건부 Euler
│   ├── train.py, sample.py     # 학습, 오프라인 생성 검사
│   ├── eval_pushing.py         # Pushing 시뮬레이터 평가 (--postprocess lipo로 ⑤만 적용 가능)
│   ├── eval_inserting.py       # Inserting 시뮬레이터 평가
│   ├── eval_task.py            # 추가 태스크(avoiding·aligning·sorting·stacking) baseline 평가
│   ├── contexts/               # Inserting 평가 초기 배치 82개, sorting context
│   └── results/
│       ├── pushing_baseline/   # baseline_ema.pt(학습된 가중치), config.json, normalizer.npz, metrics.csv
│       └── inserting_baseline/ # 위와 같은 구성
├── fm_constraint/              # train-free 5개 방법, D3IL 평가 루프, 32조합 runner
├── configs/                    # 5개 방법·제약 설정 (schema v4, pushing/inserting.yaml)
├── ood/                        # OOD 조건 정의·생성·검증·지표
├── scripts/run_experiment.sh   # 본실험: 맡은 context × 원하는 조합 순차 실행
├── experiments/id_v1/          # 본실험 결과 (git 추적)
├── visualizer/                 # 로컬 3D rollout viewer
├── preprocess/                 # data/를 만든 전처리·검증 스크립트
├── tests/, fm_constraint/tests/
└── requirements.txt            # 학습·평가에 필요한 패키지
```

**학습된 베이스라인 가중치는 `fm_baseline/results/<태스크>_baseline/baseline_ema.pt`에 있다.** 재개용 전체 체크포인트(`last.pt`, `ckpt_*.pt`)와 학습·평가 실행 폴더(`fm_baseline/runs/`)는 용량 때문에 저장소에 넣지 않는다. 6절을 참고한다.

## 2. 베이스라인 성능

| | Pushing | Inserting |
|---|---|---|
| 모델 | dim 32, dim_mults (1, 4, 8) | dim 64, dim_mults (1, 2, 4, 8) |
| 평가 | 공식 테스트 context 30개 × 4회 | 검증 분할 초기 배치 82개 × 4회 |
| 성공 기준 | 블록 2개가 목표 5cm 이내, 400스텝 | 블록 3개가 각 목표 1cm 이내, 2,000스텝 |
| **성공률** | **90.8%** | **81.4%** |
| 엔트로피 / 점수 | 0.491 / 0.700 | 0.351 / 0.583 |
| 평균 거리 | 0.044m | 0.033m |

제약 실험 결과는 **같은 태스크의 이 값과** 비교한다. 두 태스크는 난이도가 달라서 서로의 성공률을 비교하지 않는다.

## 3. 모델 개요

- 백본과 CFM은 `fm_baseline/models/`의 `TemporalUnet`, `FlowMatcher`를 사용한다. HardFlow `d3il` 커밋 `a85105080dd0a28dca76b49ac0414f275de146ed`에서 필요한 구성만 내부로 이관했으며, 기존 EMA checkpoint의 파라미터 이름·연산·학습 RNG 순서를 유지한다. 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 기록했다.
- 모델은 길이 16의 궤적 `x ∈ R^{16 × (2 + obs_dim)}`를 생성한다. 각 시점은 `[행동 2개, 관측 obs_dim개]` 순서다.
- 첫 시점(t=0)의 관측을 현재 관측으로 고정한 상태에서 노이즈를 Euler ODE 10스텝으로 적분해 궤적을 만든다.
- **모델은 정규화 공간에서 동작한다.** 원래 단위 값은 `raw = x * scale + center`이고, `center`와 `scale`은 실행 폴더의 `normalizer.npz`에 있다(길이 `2 + obs_dim`, 위 순서와 같음).

### 행동과 관측 정의

행동은 **목표 손 XY의 변화량**이다(`des_c_pos[t+1, :2] - des_c_pos[t, :2]`, 단위 m). 축마다 거의 {−0.00833, 0, +0.00833} 값만 갖는다.

| 인덱스 (관측 기준) | Pushing (12차원) | Inserting (16차원) |
|---|---|---|
| 0:2 | 목표 손 XY | 목표 손 XY |
| 2:4 | 실제 손 XY | 실제 손 XY |
| 4:8 | 빨간 블록 X, Y, sin(yaw), cos(yaw) | box-1 X, Y, sin(yaw), cos(yaw) |
| 8:12 | 초록 블록 X, Y, sin(yaw), cos(yaw) | box-2 X, Y, sin(yaw), cos(yaw) |
| 12:16 | – | box-3 X, Y, sin(yaw), cos(yaw) |

전체 궤적 `x`에서는 행동이 0:2, 관측이 2부터 시작한다. 예를 들어 Pushing 궤적의 목표 손 XY는 `x[:, 2:4]`다.

## 4. 평가 프로토콜 (모든 제약 실험에 동일하게 적용)

1. 장면에서 최신 상태를 직접 읽어 위 형식으로 관측을 만든다.
2. 16스텝 궤적을 생성하고 **앞 8스텝의 행동을 연달아 실행**한 뒤 다시 생성한다.
3. 행동을 이전 목표 손 위치에 더해 `[x, y, 고정 z, 0, 1, 0, 0]`으로 환경에 넘긴다. 첫 목표 위치는 리셋 직후의 실제 손 위치다.
4. 성공 여부, 모드, 평균 거리는 D3IL 공식 환경이 판정한다.

ODE 10스텝, 8스텝 실행, context 구성, context당 4회, 시드는 베이스라인과 같게 유지해야 비교가 된다. 바꾸면 베이스라인도 같은 조건으로 다시 평가한다.

## 5. 환경 준비 (새 VESSL 워크스페이스)

각자 자기 워크스페이스에 아래 구조를 만든다. 스크립트는 이 배치를 기본값으로 찾으므로 경로 옵션 없이 돌아간다.

```text
/root/d3il-work/
├── yflow-d3il/    # 이 저장소
├── d3il/          # D3IL 공식 저장소 (시뮬레이터, Pushing 테스트 context)
└── venv/
```

```bash
mkdir -p /root/d3il-work && cd /root/d3il-work
git clone https://github.com/DorkShiba/yflow-d3il.git
git clone https://github.com/ALRhub/d3il.git
git -C d3il checkout 1d9c71850d4fc1477cee17b557c8b97d70b13071
python3 -m venv venv
source venv/bin/activate
pip install -r yflow-d3il/requirements.txt
```

이후 터미널을 열 때마다 `source /root/d3il-work/venv/bin/activate`만 하면 된다.

- **Python 3.11** 환경을 사용한다. `requirements.txt`는 현재 검증한 Linux x86_64 환경의 패키지 버전을 고정하며, 테스트용 pytest와 PNG/GIF용 Pillow도 포함한다.
- 외부 모델 패키지는 필요하지 않다. D3IL은 checkout하여 경로로 사용하며 `pip install`하지 않는다. D3IL 모델 폴더는 시뮬레이터의 임시 XML 생성용 쓰기 권한이 필요하다.
- D3IL 데이터 ZIP(약 17GB)은 받지 않아도 된다. 학습 데이터는 `data/`에, Inserting 평가 초기 배치는 `fm_baseline/contexts/`에, Pushing 테스트 context는 D3IL 저장소에 들어 있다.
- 확인된 버전: PyTorch 2.3.1, NumPy 1.26.4, Gym 0.23.1, MuJoCo 2.3.2, Pinocchio 2.6.21, einops 0.8.1. NumPy 2.x가 설치되면 안 된다.
- `ImportError: libGL.so.1`이 나면 `apt-get update && apt-get install -y libgl1 libglib2.0-0`을 실행한다. 시스템 패키지는 워크스페이스를 재시작하면 사라진다.

설치 확인 (수 분):

```bash
cd /root/d3il-work/yflow-d3il/fm_baseline
python sample.py --run-dir results/pushing_baseline --ckpt baseline_ema.pt --no-save
python eval_inserting.py --run-dir results/inserting_baseline --ckpt baseline_ema.pt --check-obs 5
python eval_pushing.py --run-dir results/pushing_baseline --ckpt baseline_ema.pt --n-contexts 3 --execute-steps 8 --tag setup_check
```

첫 명령은 `CHECK PASSED`, 두 번째는 `check summary  max`가 1e-4 이하, 세 번째는 대부분 `success 1`이 나오면 정상이다. Pushing의 `--check-obs`는 원본 시연 pkl이 있어야 해서 새 워크스페이스에서는 쓰지 않는다.

모델 코드는 항상 저장소 내부에서 import한다. 데이터와 D3IL은 저장된 경로 또는 저장소 기준 `<저장소>/data/...`, `<저장소>/../d3il`을 사용한다. 통합 설정은 `paths.data_root`, `paths.d3il_root`로 지정한다. Baseline CLI는 `--data-dir`, `--d3il-root`(학습은 `--data-root`)를 지원한다.

## 6. 베이스라인 체크포인트

| 태스크 | 저장소 안 위치 | 크기 | 원본 (VESSL `/root/d3il-work/codex/fm_baseline/`) |
|---|---|---|---|
| Pushing | `fm_baseline/results/pushing_baseline/` | 약 15MB | `runs/pushing_main_nosn_20261014T010928Z` |
| Inserting | `fm_baseline/results/inserting_baseline/` | 약 63MB | `runs/inserting_abl_d64_nosn_20261014T105353Z` |

`baseline_ema.pt`는 원본 `last.pt`(100만 스텝)에서 **EMA 가중치만** 꺼내 저장한 파일이다(`{"ema", "step", "config"}`). 평가와 추론 시점 제약 실험에는 이것만 있으면 된다. 같은 폴더의 `config.json`과 `normalizer.npz`가 함께 있어야 불러올 수 있으므로 폴더 단위로 다룬다.

`data_dir`가 없는 경로를 가리키면 저장소의 데이터를 찾는다. 원본 checkpoint·normalizer·설정 파일은 수정하지 않아도 된다. 학습을 이어서 하려면 옵티마이저 상태가 든 원본 `last.pt`가 필요하다.

## 7. 불러와서 쓰기

```python
import sys
sys.path.insert(0, "/root/d3il-work/yflow-d3il/fm_baseline")
import numpy as np
from eval_pushing import load_model, Policy

run = "/root/d3il-work/yflow-d3il/fm_baseline/results/pushing_baseline"
cfg, model, norm, apply_conditioning, Solver, step = load_model(run, "baseline_ema.pt", "ema", device="cuda:0")
policy = Policy(model, norm, cfg, apply_conditioning, Solver, ode_steps=10, device="cuda:0")

obs = np.zeros(cfg["obs_dim"], dtype=np.float32)
plan = policy.plan(obs)
actions = plan[:, :2]
```

`plan`은 원래 단위의 `(16, 2 + obs_dim)` 배열이다. `load_model`은 `config.json`을 읽어 모델 크기를 자동으로 맞춘다.

## 8. train-free 5개 방법

| # | 방법 | 적용 위치 | CLI |
|---|---|---|---|
| ① | Projection | Predict 매 Euler 직후, 행동 공동 투영 | `--projection` |
| ② | PF2MP | Predict 벡터장, train KDE potential(상태 위치) | `--pf2mp` |
| ③ | Filter | 실행 직전 최신 관측으로 행동 하나 투영, 수정 시 재계획 | `--filter` |
| ④ | RTC | Predict 벡터장, 직전 계획의 미실행 행동 inpainting | `--rtc` |
| ⑤ | LiPo | 계획 끝 고정 한도 jerk QP(①이 켜져 있으면 ① 제약 포함) | `--lipo` |

```bash
MUJOCO_GL=disable python -m fm_constraint.eval_pushing --config configs/pushing.yaml --projection --rtc --lipo
MUJOCO_GL=disable python -m fm_constraint.eval_inserting --config configs/inserting.yaml --pf2mp --filter --ood D_mid
bash scripts/run_experiment.sh --name <이름> --contexts 0-6   # 본실험 (아래 "실험 분담")
```

정의와 상호작용 규칙은 [`docs/METHODS.md`](docs/METHODS.md)에 있다. 모든 방법을 끄면 베이스라인과 같은 계획을 만든다. 평가 프로토콜(4절)은 바꾸지 않는다.

### 베이스라인 실패 유형 (제약이 겨냥할 수 있는 지점)

- Pushing 실패 11/120: 목표 근처까지 밀었지만 5cm 안에 못 넣음 7건, 목표 손 위치가 실제 손에서 24~30cm까지 벌어짐 3건.
- Inserting 실패 61/328: 손이 멈춤 34건, 목표–손 거리 10cm 초과 30건(미로 벽에 막힌 상태로 목표 위치만 이동하는 것으로 추정), 평균 거리 2cm 미만의 근소한 실패 16건.

baseline 평가 폴더의 `rollouts.npz`에는 rollout마다 목표 손 위치(`*_des`), 실제 손 위치(`*_c`), 블록 상태(`*_boxes`), 실행 행동(`*_act`)이, `fm_constraint` 평가의 `rollouts.json`에는 step별 실행 행동·제약 잔차·계획 보고가 저장된다.

## 9. 자주 하는 명령

```bash
source /root/d3il-work/venv/bin/activate
cd /root/d3il-work/yflow-d3il/fm_baseline
P=results/pushing_baseline; I=results/inserting_baseline
python sample.py --run-dir $P --ckpt baseline_ema.pt
python eval_pushing.py --run-dir $P --ckpt baseline_ema.pt --n-contexts 30 --n-traj 4 --execute-steps 8 --tag <이름>
python eval_inserting.py --run-dir $I --ckpt baseline_ema.pt --n-traj 2 --seed 0 --tag <이름>_s0
python eval_inserting.py --run-dir $I --ckpt baseline_ema.pt --n-traj 2 --seed 1 --tag <이름>_s1
```

평가 결과는 `--run-dir` 아래 `eval_*` 폴더에, 새로 학습한 실행 폴더는 `fm_baseline/runs/`에 저장되며 둘 다 git에서 제외된다. 평가 시간은 Pushing 120회 약 25분, Inserting 82 × 2회 약 3.3시간(시드 두 개를 동시에 실행)이다. 학습 명령과 옵션은 `fm_baseline/README.md` 7절에 있다.

## 10. 테스트와 viewer

```bash
python -m pytest fm_constraint/tests tests -q     # visualizer 테스트는 visualizer/requirements.txt 필요
./visualizer/run.sh                                # 로컬 3D rollout viewer (visualizer/README.md)
```

## 11. 실험 분담 (ID 본실험)

프로토콜은 [`docs/report.md`](docs/report.md)에 있다. 태스크마다 30개 context(`configs/experiment_contexts.json`, 번호 0~29)를 고정해 두었고, context마다 FM 입력 noise seed 0·1·2로 3회 실행한다. **Pushing과 Inserting은 같은 번호를 쓴다.** 각자 맡은 context에 대해 32조합을 모두 실행하므로, 같은 context의 조합 비교는 항상 같은 장비에서 이루어진다.

| 담당 | context 번호 | 개수 | 실행 명령 |
|---|---|---:|---|
| 형빈 | 0–6 | 7 | `bash scripts/run_experiment.sh --name hyeongbin --contexts 0-6` |
| 준혁 | 7–16 | 10 | `bash scripts/run_experiment.sh --name junhyuk --contexts 7-16` |
| 재엽 | 17–19 | 3 | `bash scripts/run_experiment.sh --name jaeyeop --contexts 17-19` |
| 성민 | 20–29 | 10 | `bash scripts/run_experiment.sh --name seongmin --contexts 20-29` |

- **같은 실험 보장:** 스크립트는 다음을 확인한 뒤에만 실행한다.
  - 커밋하지 않은 코드 변경이 없다.
  - 요청한 GPU에서 실제로 계산이 된다(쓸 수 없으면 CPU로 넘어가지 않고 멈춘다).
  - D3IL에서 읽은 초기 상태가 목록 파일과 같다(D3IL은 커밋 `1d9c718`로 맞춘다).
  - noise는 장비와 무관하게 같은 값이 생성된다.
- **나눠서 실행:** 맡은 범위를 여러 번에 나눠도 된다. 예: `--contexts 0-3`과 `--contexts 4-6`을 다른 GPU(`--device cuda:1`)나 다른 날 실행. 태스크만(`--task inserting`) 또는 조합 일부만(`--combos 16-31`) 실행할 수도 있다. 단, **같은 context의 32조합은 같은 장비에서** 끝낸다.
- **중단·재실행:** 같은 명령을 다시 실행하면 완료된 단위(task × 조합 × context)는 건너뛰고, 중단된 단위만 처음부터 다시 실행한다.
- **동작 확인:** `--smoke 24`(24 step만, 결과는 로컬에만 저장)와 `--dry-run`을 먼저 실행한다.
- **결과 공유:**
  - 끝나면 `docs/report_<이름>.md`가 생성된다.
  - `experiments/id_v1/`(단위마다 `result.json`과 작은 `trajectories.npz`)와 보고서를 commit해서 push한다. 스크립트가 마지막에 명령을 출력한다.
  - 전체 계획 로그(`rollouts.json`, Inserting rollout 1개에 최대 약 8 MB)는 `fm_constraint/runs/experiment_id_v1/full/`에 로컬로만 남는다.
- **통합(팀장):** 모든 결과가 push되면 `python -m fm_constraint.experiment merge`로 30 context 전체의 `추정값 [95% CI]`와 Complementarity Δ를 `docs/report_results.md`에 만든 뒤 `docs/report.md`에 옮긴다. 코드 버전이 섞였거나 누락된 context가 있으면 표시하거나 멈춘다.
- **예상 시간(baseline 기준, 방법을 켜면 더 길다):** Pushing은 context 1개 × 32조합에 약 20분, Inserting은 약 4시간이다. 예를 들어 10개를 맡으면 Inserting에 약 40시간이 걸린다. 첫 실행 로그의 단위별 시간으로 다시 계산한다.
