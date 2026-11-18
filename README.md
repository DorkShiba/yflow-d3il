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
├── scripts/run_experiment.sh   # 본실험(ID·OOD): 맡은 context × 원하는 조합 순차 실행
├── scripts/run_stacking_experiment.sh  # Stacking 본실험(ID·OOD A), 11절 "Stacking 분담"
├── scripts/run_wall_experiment.sh      # 벽 OOD 본실험(Pushing·Inserting), 11절 "벽 OOD 분담"
├── experiments/                # 본실험 단위 결과 (로컬 전용, git 제외. 공유는 docs/report_*.md)
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

## 11. 실험 분담 (ID·OOD 본실험)

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
- **OOD 실험:** 같은 분담·같은 명령에 `--ood <조건>`만 붙인다. 예: `bash scripts/run_experiment.sh --name hyeongbin --contexts 0-6 --ood A_mid`.
  - ID와 같은 30 context, 같은 noise seed 0·1·2, **같은 FM noise**를 쓰고 OOD 조건(A_mid는 블록 초기 위치)만 다르다. 그래서 ID 결과와 context·seed 단위로 짝지어 비교할 수 있다.
  - 조건 정의는 [`ood/README.md`](ood/README.md), 사례는 `data/ood/`(commit된 파일, sha256을 보고서에 기록)에 있다.
- **OOD A_weak·A_strong 분담 (2026-11-12, A 수치 1/5/18 cm, `data/ood` v3):** A_mid는 다시 실행하지 않는다. 30 context를 네 명이 같은 양(8·8·7·7)으로 나누고, 각자 맡은 context에서 **A_weak와 A_strong을 같은 장비로** 둘 다 실행한다(강도 간 비교도 같은 장비에서 이루어지게). 한 명령이 A_weak → A_strong 순서로 돌고, 조건마다 보고서를 만든다.

  | 담당 | context 번호 | 개수 | 실행 명령 |
  |---|---|---:|---|
  | 형빈 | 0–7 | 8 | `bash scripts/run_experiment.sh --name hyeongbin --contexts 0-7 --ood A_weak,A_strong` |
  | 준혁 | 8–15 | 8 | `bash scripts/run_experiment.sh --name junhyuk --contexts 8-15 --ood A_weak,A_strong` |
  | 재엽 | 16–22 | 7 | `bash scripts/run_experiment.sh --name jaeyeop --contexts 16-22 --ood A_weak,A_strong` |
  | 성민 | 23–29 | 7 | `bash scripts/run_experiment.sh --name seongmin --contexts 23-29 --ood A_weak,A_strong` |

  - 올릴 보고서: `docs/report_ood_A_weak_<이름>.md`, `docs/report_ood_A_strong_<이름>.md`. 통합은 `python -m fm_constraint.experiment merge --ood A_weak`(A_strong도 같다).
  - Pushing A_strong의 context 13(test_028)은 17 cm, context 29(test_059)은 8 cm로 낮춰져 있다(배치 불가·시뮬레이터 안착 실패, `shift[].shift_m`). 분석할 때 따로 표시한다.
- **추가 A 강도 분담 (2026-11-17, Pushing·Inserting A_10, Stacking A_3):** A_strong(18 cm)에서 Inserting은 32조합 모두 성공 0%였고, Stacking은 A_mid(5 cm)에서도 성공률이 2–9%였다. 그래서 Pushing·Inserting에 **A_10**(시연 범위 밖 10 cm), Stacking에 **A_3**(3 cm)를 더한다. 준혁 GPU를 쓸 수 없어 context 대신 **태스크 단위로** 나눈다. 한 사람이 한 태스크의 30 context × 32조합(960 단위, 2,880 rollout)을 모두 같은 장비에서 실행한다.

  | 담당 | 태스크 · 조건 | 실행 명령 | 올릴 보고서 |
  |---|---|---|---|
  | 형빈 | Inserting A_10 | `bash scripts/run_ood_extra.sh --member hyeongbin --workers 6` | `docs/report_ood_A_10_hyeongbin.md` |
  | 재엽 | Pushing A_10 | `bash scripts/run_ood_extra.sh --member jaeyeop --workers 4` | `docs/report_ood_A_10_jaeyeop.md` |
  | 성민 | Stacking A_3 | `bash scripts/run_ood_extra.sh --member seongmin --workers 4` | `docs/stacking/report_A_3_seongmin.md` |

  ```bash
  git pull                                   # 코드와 data/ood(A_10 행 추가), data/ood_stacking(A_3 추가) 받기
  # 1) 동작 확인: 16 step만, 결과는 로컬(fm_constraint/runs)에만, 보고서 없음
  bash scripts/run_ood_extra.sh --member <이름> --smoke 16 --contexts 0 --combos 0,31 --workers 2
  # 2) 본실험: 끝나면 보고서를 만들고 git 명령을 출력한다. 중단되면 같은 명령으로 이어서 실행한다
  bash scripts/run_ood_extra.sh --member <이름> --workers <N> [--device cuda:1]
  # 진행 확인(다른 터미널)
  tail -f fm_constraint/runs/experiment_logs/<이름>_*_w1_*.log
  # 3) 보고서만 올린다
  git add <보고서> && git commit -m "report: <이름> <태스크> <조건> contexts 0-29" && git pull --rebase && git push
  ```

  - **worker:** 같은 GPU에 프로세스를 N개 띄운다. 모두 같은 단위 목록을 돌고, 단위마다 잠금을 잡으며, 결과가 있으면 건너뛴다(`BUSY`·`SKIP` 로그는 정상이다). 시뮬레이터가 CPU를 쓰므로 N은 CPU 코어 수 이하로 둔다. 보고서는 모든 worker가 끝난 뒤 한 번 만든다. `--dry-run`은 검사만 하고 실행할 명령을 출력한다.
  - **예상 시간**(A_strong 보고서의 단위당 시간 기준, 장비에 따라 다름): Pushing 단위당 약 1분 → 직렬 약 16시간, Inserting 단위당 약 5분 → 직렬 약 80시간, Stacking 단위당 약 1.3분 → 직렬 약 20시간. worker 수만큼 대략 나뉜다. Inserting이 가장 길어서 worker를 가장 많이 둔다.
  - **스크립트가 시작 전에 확인하는 것:** OOD 파일 해시가 manifest와 같은지(`git pull`을 안 했으면 여기서 멈춘다), 커밋하지 않은 코드 변경이 없는지, GPU에서 실제로 계산이 되는지.
  - **데이터:** `data/ood/*.jsonl` 끝에 A_10 행만 덧붙였다(`python -m ood.generate --add A_10`). 기존 20조건 행은 바이트 단위로 같다. Stacking은 `stacking.jsonl` 끝에 A_3 행을 덧붙이고 `stacking_A_3_contexts.pkl`을 새로 만들었다(`python -m ood.generate_stacking --add A_3`). 기존 pkl과 `max_shift.json`은 그대로다. 두 manifest의 `additions`에 기록이 있다.
  - **배치 규칙:** A_10은 5 cm보다 크므로 A_strong과 같은 규칙(팔 도달 고리, 다른 블록 시작 범위와 5 cm, 2 s 안착 확인, 안 되면 그 context만 1 cm씩 낮춤)을 쓴다. 실험 30 context 중 Inserting은 **13개가 9 cm, 1개(context 20)가 6 cm**로, Pushing은 1개(context 29, test_059)가 8 cm로 낮아졌다. 실제 d는 `shift[].shift_m`에 있다. A_3은 A_weak·A_mid와 같은 규칙이며 실험 30 context에서 세 블록이 모두 옮겨졌다(100개 중 1개만 2개 이동).
  - **제약 적용 불가 context (2026-11-17 결정: 따로 보고):** ①투영·③필터의 블록 중심 8각형 제약(`object_envelope`, Pushing 여유 0.25 m)은 매 계획마다 **현재 블록 위치**를 중심으로 만든다. 그래서 블록이 멀리 옮겨지면 고정 시작 손이 다각형 밖에 남는다. 첫 행동은 가속도 제약 때문에 β(Pushing 1.67 cm, Inserting 1.73 cm)까지만 움직일 수 있다. 따라서 손이 다각형 밖으로 β보다 멀면 ①·③ 조합은 첫 계획부터 실행 불가능해 방법 실패로 끝난다. 방법 성능이 아니라 제약이 그 배치에 적용되지 않는 경우이므로, **주 비교는 적용 가능 context로 하고 적용 불가 context는 따로 보고한다.**
    - A_strong Pushing: 적용 불가 10개(4, 5, 6, 8, 9, 12, 14, 21, 22, 27)에서 ①·③ 조합의 방법 실패가 100%, 나머지 20개에서는 0%였다. 10개 모두 두 블록이 목표 쪽(y+)으로 옮겨진 쉬운 배치다(P00 성공 93%, 나머지 17%). 결과는 [`docs/report_ood_A_strong_envelope_split.md`](docs/report_ood_A_strong_envelope_split.md)에 있다. A_mid context 12(1.64 cm 밖, β 미만)는 정상 실행돼 기준과 맞는다.
    - A_10 예측: Pushing 적용 불가 **10개**(3, 4, 5, 6, 8, 12, 14, 22, 24, 27; context 6은 1.68 cm로 경계), 9·21은 밖이지만 β 미만이다. Inserting은 1개(context 20, 2.6 cm)다.
    - 실험이 끝나면 `python -m ood.envelope_split --cond A_10`으로 같은 표(`docs/report_ood_A_10_envelope_split.{md,json}`)를 만든다. 보고서만 읽으므로 merge처럼 누구나 실행할 수 있다.
  - **이동 방향:** A_10·A_strong 규칙은 팔 도달 고리 때문에 주로 y±로 옮긴다(Pushing A_10 블록 60개 중 y+ 39개). y+는 목표 쪽이라 쉬워지므로 결과를 방향별로 나눠 본다(보고서 원시 데이터의 OOD case → `shift[].face`).
  - **code fingerprint:** Pushing·Inserting runner의 fingerprint는 현재 `e55b2836ed46`로, A_weak·A_strong 보고서의 `848e5eb61481`과 다르다. 그 사이 팀원 sorting 커밋(`b22441b`, `4dd0b2d`)이 `configs/config.py`·`fm_baseline/train.py`·`fm_constraint/pf2mp.py`를 바꿨기 때문이다. sorting 분기만 추가돼 Pushing·Inserting 동작에는 영향이 없다. Stacking runner는 조건 목록에 A_3을 넣어 `ae2ad005…`에서 `2206f1b7…`로 바뀌었다. 통합은 조건별로 하므로 A_10·A_3 통합은 문제없다. 다만 **이전 조건(A_weak 등)의 미완료 단위를 지금 코드로 이어서 실행할 수는 없다**(fingerprint가 다르면 runner가 멈춘다).
  - **통합:** `python -m fm_constraint.experiment merge --ood A_10`(보고서 2개: Pushing 재엽, Inserting 형빈), `python -m fm_constraint.stacking_experiment merge --cond A_3`.
- **결과 공유 — 보고서만 git에 올린다:**
  - 단위 결과는 **로컬에만** 남는다(`.gitignore`): `experiments/id_v1/`, `experiments/ood_v1/<조건>/`(단위마다 `result.json`, `trajectories.npz`), 전체 계획 로그 `fm_constraint/runs/experiment_<protocol>/full/`(`rollouts.json`, Inserting rollout 1개에 최대 약 8 MB). 다시 만들려면 재실행해야 하므로 지우지 말고 필요하면 각자 백업한다.
  - 실행이 끝나면 스크립트가 보고서를 만든다. ID는 `docs/report_<이름>.md`, OOD는 `docs/report_ood_<조건>_<이름>.md`. 결과를 나중에 더 실행했다면 `python -m fm_constraint.experiment report --name <이름> [--ood <조건>]`로 다시 만든다(덮어쓴다).
  - **그 보고서 파일 하나만** commit해서 push한다. 스크립트가 마지막에 명령을 출력한다.
  - 이름 `results`와 `ood`로 시작하는 이름은 통합 보고서와 겹치므로 쓸 수 없다.

### Stacking 분담 (ID·OOD A, 2026-11-16)

Stacking도 32조합을 **ID, A_weak, A_mid, A_strong** 네 조건에서 비교한다. (아래 끝 문단의 [팀원 실행 안내](docs/stacking_team_run.md)는 베이스라인 대 조합 31만 보는 별도 ID 비교다. 본실험 분담은 이 절을 따른다.) 모델은 재엽이 학습한 관절 공간 모델(`fm_baseline/results/stacking_baseline`)이고, 방법은 같은 5개를 7관절 공간으로 옮긴 버전이다(④ RTC 포함, 그리퍼는 수정하지 않음). context는 Pushing·Inserting과 **다른** 30개(`configs/stacking_experiment_contexts.json`)를 쓴다. Stacking test 0–99에서 고정 seed로 뽑았다. 각 context를 noise seed 0·1·2로 3회 실행한다. 정의·지표·검증은 [`docs/stacking/README.md`](docs/stacking/README.md)에 있다.

각자 맡은 context에서 **네 조건 × 32조합을 같은 장비로** 실행한다. 한 명령이 id → A_weak → A_mid → A_strong 순서로 돌고, 조건이 끝날 때마다 보고서를 만든다.

| 담당 | context 번호 | 개수 | 실행 명령 |
|---|---|---:|---|
| 형빈 | 0–7 | 8 | `bash scripts/run_stacking_experiment.sh --name hyeongbin --contexts 0-7` |
| 준혁 | 8–15 | 8 | `bash scripts/run_stacking_experiment.sh --name junhyuk --contexts 8-15` |
| 재엽 | 16–22 | 7 | `bash scripts/run_stacking_experiment.sh --name jaeyeop --contexts 16-22` |
| 성민 | 23–29 | 7 | `bash scripts/run_stacking_experiment.sh --name seongmin --contexts 23-29` |

- **준비:** `git pull`만 하면 된다. 모델, PF 캐시(`joint_pf_cache.npz`, sha256 고정), OOD 배치(`data/ood_stacking/`)가 모두 저장소에 있다. 학습 시연 데이터는 필요 없다. D3IL은 기존과 같이 `../d3il`(커밋 `1d9c718`)을 쓴다.
- **동작 확인 먼저:** `bash scripts/run_stacking_experiment.sh --name <이름> --contexts <맡은 첫 번호> --cond id --combos 0,29 --smoke 40`. 40 step만 돌고 결과는 로컬에만 저장된다. `--dry-run`은 실행할 명령만 출력한다.
- **같은 실험 보장:** 다음 중 하나라도 어긋나면 시작하지 않는다.
  - 커밋하지 않은 코드 변경이 있다.
  - 요청한 GPU를 쓸 수 없다.
  - PF 캐시 sha256이 고정값과 다르다.
  - 초기 상태가 목록 파일과 다르다.
  - OOD 파일이 manifest와 다르다.
- **나눠서 실행·중단:** 기존과 같다. `--contexts 0-3`과 `4-7`을 다른 GPU(`--device cuda:1`)나 다른 날 실행해도 된다. 조건만(`--cond A_strong`), 조합 일부만(`--combos 16-31`) 실행할 수도 있다. 같은 명령을 다시 실행하면 완료된 단위(조건 × 조합 × context)는 건너뛴다. 시뮬레이터가 CPU 시간 대부분을 쓰므로 GPU 하나에 프로세스를 2–4개 띄워도 된다. 단, **같은 context는 모든 조건·조합을 같은 장비에서** 끝낸다. GPU가 다르면 같은 seed라도 수치가 조금 달라진다.
- **예상 시간:** context 1개 × 4조건 × 32조합 = 384 rollout이다. 3090 기준 rollout당 약 25초라 context 1개에 약 2.7시간, 8개면 약 21시간이다. 첫 실행 로그의 단위별 시간으로 다시 계산한다.
- **결과 공유:** 단위 결과(`experiments/stacking_v1/`)는 로컬에만 남는다. 스크립트가 만드는 보고서 `docs/stacking/report_<조건>_<이름>.md` 4개만 commit해서 push한다. 마지막에 명령이 출력된다. 보고서를 다시 만들려면 `python -m fm_constraint.stacking_experiment report --cond <조건> --name <이름>`.
- **통합(누구나):** `python -m fm_constraint.stacking_experiment merge --cond <조건>` → `docs/stacking/report_<조건>_results.{md,json}`. 조합별 추정값 [95% CI], Complementarity Δ, **목표 진입 1개·2개 비율**, 관절 기준 위반율에 더해, A 조건에서는 옮겨진 블록 수(1·2·3개)별 성공률도 낸다. A_strong에서는 옮길 수 있는 블록만 18 cm 옮겼다. 30개 중 1개만 옮겨진 context가 18개, 2개가 8개, 3개가 4개다.

### 벽 OOD 분담 (Pushing·Inserting, 2026-11-18)

학습 때 없던 **정적 벽**을 통로 중앙(x = 0.525)에 둔다. 모델은 벽을 관측하지 못한다. 벽 회피 부등식은 충돌 제약 방법 중 QP가 있는 **①투영과 ③필터**에만 넣는다. ⑤LiPo는 기존 규칙대로 ①이 켜졌을 때만 받고, ②PF2MP에는 넣지 않는다. ID와 같은 30 context × noise 3 × 32조합으로 실행하며, 장면에 벽이 있다는 것만 다르다. 태스크 단위로 나눈다.

| 담당 | 태스크 | 벽 | 실행 명령 | 올릴 보고서 |
|---|---|---|---|---|
| 형빈 | Pushing | 세로 2×10 cm, 중심 (0.525, 0.15) | `bash scripts/run_wall_experiment.sh --name hyeongbin --task pushing --workers 6` | `docs/report_wall_hyeongbin.md`, `docs/wall_summary_pushing_hyeongbin.md` |
| 성민 | Inserting | 기둥 2×2 cm, 게이트 중앙 (0.525, 0.17) | `bash scripts/run_wall_experiment.sh --name seongmin --task inserting --workers 4` | `docs/report_wall_seongmin.md`, `docs/wall_summary_inserting_seongmin.md` |

```bash
git pull
# 1) 동작 확인: 40 step만, 결과는 로컬(fm_constraint/runs/experiment_wall_v1/smoke)에만, 보고서 없음
bash scripts/run_wall_experiment.sh --name <이름> --task <태스크> --smoke 40 --contexts 0 --combos 0,16 --workers 2
# 2) 본실험: 끝나면 보고서와 벽 지표 요약을 만들고 git 명령을 출력한다. 중단되면 같은 명령으로 이어서 실행한다
bash scripts/run_wall_experiment.sh --name <이름> --task <태스크> --workers <N> [--device cuda:1]
# 진행 확인(다른 터미널)
tail -f fm_constraint/runs/experiment_logs/<이름>_<태스크>_wall_*_w1_*.log
# 3) 보고서 2개만 올린다
git add docs/report_wall_<이름>.md docs/wall_summary_<태스크>_<이름>.md && git commit -m "report: <이름> <태스크> wall" && git pull --rebase && git push
```

- **벽 제약 (`fm_constraint/wall.py`):** 손에만 건다. 벽 상자를 rod 반지름 1 cm + 여유 1.2 cm만큼 키운다. 블록은 키우지 않는다. 계획의 각 시점 h마다 반평면 하나를 둔다. 반평면은 그 시점 nominal 손 위치에서 가장 가까운 벽 면(부호 거리 최대)으로 고르고, 매 Euler 단계마다 다시 고른다. 점이 키운 벽 안에 있으면 직전 시점의 면을 유지한다(벽을 뚫고 지나가는 계획 방지). ③은 실행 직전 nominal로 면을 고른다. 벽의 위치·크기·여유는 결과마다 `experiment_metadata.wall`에 기록된다.
- **추가 지표 (episode마다, `summary`가 표로 만든다):** 손–벽 접촉, 블록–벽 접촉(시뮬레이터 접촉), 명령 벽 침범, 벽 앞 정지(40 step 동안 1 cm 안에서만 움직이고 벽에서 3 cm 이내), 최소 손–벽 거리. 다시 만들려면 `python -m fm_constraint.wall_experiment summary --task <태스크> --out docs/wall_summary_<태스크>_<이름>.md`.
- **smoke 결과 (형빈, 벽을 지나는 context만):** ①·③ 조합은 명령 벽 침범이 0%였다(벽 제약 없는 P00 78%, I00 100%). 다만 추종 오차 때문에 실제 손이 1–6 step 동안 0–5 mm 닿는 경우가 남는다. 여유를 6 mm로 줄이면 이런 접촉이 두 배가 되어 1.2 cm로 정했다.
- **예상 시간 (smoke 기준, 장비에 따라 다름):** Pushing 단위당 약 3.5분이다. 960 단위면 직렬 약 55시간, worker 6개로 약 10시간이다. Inserting은 벽 앞에서 오래 머무는 rollout이 많아 단위당 약 10–16분이다. 직렬 160–250시간, worker 수만큼 나뉜다. 그러니 worker는 CPU 코어 수 안에서 최대로 둔다.
- **같은 실험 보장:** 다음이면 시작하지 않는다.
  - 커밋하지 않은 코드 변경이 있다.
  - GPU를 쓸 수 없다.
- **fingerprint:** 기존 `CODE_FILES`에 `wall.py`·`wall_experiment.py`를 더한 값이다. 기존 평가 코드는 수정하지 않고, `wall_experiment`가 실행 시점에 클래스를 바꿔 끼운다.
- **결과 위치:** 단위 결과는 `experiments/wall_v1/`, 전체 로그는 `fm_constraint/runs/experiment_wall_v1/`에 남고, 둘 다 로컬에만 있다.
- **통합:** `python -m fm_constraint.wall_experiment merge` → `docs/report_wall_results.{md,json}`. 보고서만 읽으므로 누구나 실행할 수 있다.

### 결과 통합 (누구나 할 수 있다)

보고서 끝의 접힌 블록(`merge용 원시 데이터`)에 단위(task × 조합 × context)마다 context별 지표 7개(성공률, 위반 step, 평가 step, action change, jerk, entropy, 방법 실패율)와 code fingerprint·seed·OOD 파일 hash가 JSON으로 들어 있다. 그래서 다른 팀원의 결과 파일 없이 **보고서만으로** 30 context 전체의 CI를 계산할 수 있다. 이 블록은 손으로 고치지 않는다.

```bash
git pull                                                   # 모든 팀원의 docs/report_*.md 받기
python -m fm_constraint.experiment merge                   # ID  → docs/report_results.md, .json
python -m fm_constraint.experiment merge --ood A_mid       # OOD → docs/report_ood_A_mid_results.md, .json
git add docs/report_results.* docs/report_ood_*_results.* && git commit -m "report: merge" && git push
```

- `merge`는 `docs/report_*.md` 중 **같은 protocol**(ID는 `id_v1`, OOD는 `ood_v1_<조건>`)의 보고서만 읽는다. 파일 이름이 아니라 블록 안의 protocol로 구분하고, 통합본(`*_results.md`)은 데이터 블록이 없어 입력에서 빠진다.
- 출력: 조합별 `추정값 [95% CI]`(context cluster bootstrap 2,000회, seed 2026)와 Complementarity Δ. `.md`는 사람용, `.json`은 같은 수치와 입력 보고서 목록(`meta`)을 담는다. 둘 다 commit한다.
- 멈추는 경우: code fingerprint가 둘 이상(다른 코드로 실행한 결과 혼합), OOD 파일 hash가 둘 이상, 같은 단위가 두 보고서에 있음(분담 중복), seed가 0·1·2가 아님.
- 누락된 context가 있는 조합은 `미완료(누락 context 수)`로 표시하고 CI를 계산하지 않는다. 모두 모일 때까지 몇 번이든 다시 실행해도 된다(결과는 매번 같다).
- 최종 표는 `docs/report.md` 8절(ID)이나 OOD 절로 옮긴다.

#### ID 결과 통합 (2026-11-09 이전 방식으로 실행한 결과)

**ID 통합은 이미 끝났다**(2026-11-09, `docs/report_results.md`·`.json`, 30 context 전체, code fingerprint `0167b01b63c1`). 아래는 ID를 다시 통합해야 할 때의 절차다.

ID 본실험은 결과를 git에 올리던 때 실행해서 `experiments/id_v1/`이 저장소에 이미 commit되어 있다. `.gitignore`는 이미 추적 중인 파일에는 적용되지 않으므로 이 폴더는 그대로 남는다. 새로 생기는 OOD 결과(`experiments/ood_v1/`)만 git에서 빠진다. 그때 만든 `docs/report_<이름>.md`는 예전 형식이라 merge용 원시 데이터 블록이 없어서 **merge 입력으로 쓰이지 않는다**. 실험을 다시 돌릴 필요는 없다. 한 사람이 commit된 결과로 모든 팀원의 보고서를 다시 만든 뒤 merge한다.

```bash
git pull                                                   # experiments/id_v1/ 전체와 이 코드 받기
for n in hyeongbin junhyuk jaeyeop seongmin; do            # 결과에 기록된 member 이름
    python -m fm_constraint.experiment report --name "$n"  # docs/report_<이름>.md를 새 형식으로 덮어씀
done
python -m fm_constraint.experiment merge                   # → docs/report_results.md, .json
git add docs/report_*.md docs/report_results.json && git commit -m "report: ID merge" && git push
```

- 다시 만든 보고서의 위쪽 점추정 표는 예전과 같고, 끝에 원시 데이터 블록만 추가된다. 이 절차로 만든 merge 결과는 기존 `report_results.json`과 모든 값이 같다(2026-11-09 확인).
- 보고서에는 결과 파일에 기록된 code fingerprint가 그대로 들어간다. 모든 ID 결과가 같은 코드로 실행됐다면 merge가 통과한다.
- 한 팀원의 결과가 git에 없으면 `No results by <이름>`이 나온다. 그 팀원은 자기 로컬 결과로 같은 `report` 명령을 실행해서 보고서를 push한다.
- **예상 시간(baseline 기준, 방법을 켜면 더 길다):** Pushing은 context 1개 × 32조합에 약 20분, Inserting은 약 4시간이다. 예를 들어 10개를 맡으면 Inserting에 약 40시간이 걸린다. 첫 실행 로그의 단위별 시간으로 다시 계산한다.

Stacking을 새로 clone하여 baseline vs ①②③④⑤를 실행하려면 [팀원 실행 안내](docs/stacking_team_run.md)를 참고합니다. 모델·캐시 파일 목록, 설치, smoke와 전체 paired 비교 명령을 제공합니다. 기존 ①②③⑤ 보고 결과와 ④를 추가한 실행은 구분합니다.
