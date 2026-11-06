# Y-Flow D3IL: 플로우 매칭 베이스라인과 제약 실험

연세대학교 인공지능종합설계 Y-Flow 팀의 D3IL 실험 저장소다. 상태 기반 플로우 매칭 정책을 D3IL **Pushing**과 **Inserting** 태스크로 학습하고, 물리 제약 주입 방식들을 조합해 비교하는 것이 목표다.

현재는 **물리 제약 보정이 없는 순수 플로우 매칭 베이스라인**까지 완성됐다. 이 문서는 이 베이스라인을 불러와 제약 실험을 붙일 사람을 위한 안내다.

- 데이터 출처, 전처리, 분할: [`data/README.md`](data/README.md)
- 베이스라인 학습·평가의 상세 기록과 실패 분석: [`fm_baseline/README.md`](fm_baseline/README.md)
- 방법 ④ L2C2 기반 정책 정규화의 정의·학습·평가: [`fm_l2c2/README.md`](fm_l2c2/README.md). 학습이 필요한 FM 변형이며 `python -m fm_l2c2.train`으로 실행한다.

## 1. 저장소 구조

```text
yflow-d3il/
├── data/                       # 전처리된 상태·행동 데이터 (시연별 npz)
│   ├── pushing_sincos_v1/      # train 1,800 / val 200
│   └── inserting_sincos_v1/    # train 718 / val 82
├── fm_baseline/
│   ├── train.py                # 학습
│   ├── sample.py               # 오프라인 생성 검사
│   ├── eval_pushing.py         # Pushing 시뮬레이터 평가
│   ├── eval_inserting.py       # Inserting 시뮬레이터 평가
│   ├── contexts/
│   │   └── inserting_val.pkl   # Inserting 평가 초기 배치 82개 (원본 시연에서 추출)
│   └── results/
│       ├── pushing_baseline/   # baseline_ema.pt(학습된 가중치), config.json, normalizer.npz, metrics.csv, 평가 결과
│       └── inserting_baseline/ # 위와 같은 구성
├── preprocess/                 # data/를 만든 전처리·검증 스크립트
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

- 백본은 HardFlow 공식 저장소(`d3il` 브랜치, commit `a85105080dd0a28dca76b49ac0414f275de146ed`)의 `TemporalUnet`과 `FlowMatcher`(CFM)이며 코드를 수정하지 않고 import한다.
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
├── HardFlow/      # HardFlow d3il 브랜치 (지정 commit)
├── d3il/          # D3IL 공식 저장소 (시뮬레이터, Pushing 테스트 context)
└── venv/
```

```bash
mkdir -p /root/d3il-work && cd /root/d3il-work
git clone https://github.com/DorkShiba/yflow-d3il.git
git clone -b d3il https://github.com/azizanlab/HardFlow.git
git -C HardFlow checkout a85105080dd0a28dca76b49ac0414f275de146ed
git clone https://github.com/ALRhub/d3il.git
python3 -m venv venv
source venv/bin/activate
pip install -r yflow-d3il/requirements.txt
```

이후 터미널을 열 때마다 `source /root/d3il-work/venv/bin/activate`만 하면 된다.

- Python 3.10 또는 3.11을 쓴다. `pin`(Pinocchio)은 3.12 이상용 wheel이 없어 설치가 실패할 수 있다.
- HardFlow와 D3IL은 받기만 하고 수정하거나 `pip install` 하지 않는다. HardFlow의 `requirements.txt`는 버전이 달라 설치하지 않는다.
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

경로 결정 규칙: `config.json`에 기록된 경로가 있으면 그것을, 없으면 저장소 기준 위치(`<저장소>/data/...`, `<저장소>/../HardFlow`, `<저장소>/../d3il`)를 쓴다. 다른 배치를 쓰면 `--hardflow-root`, `--data-dir`, `--d3il-root`(학습은 `--data-root`, `--hardflow-root`)로 지정한다.

## 6. 베이스라인 체크포인트

| 태스크 | 저장소 안 위치 | 크기 | 원본 (VESSL `/root/d3il-work/codex/fm_baseline/`) |
|---|---|---|---|
| Pushing | `fm_baseline/results/pushing_baseline/` | 약 15MB | `runs/pushing_main_nosn_20261014T010928Z` |
| Inserting | `fm_baseline/results/inserting_baseline/` | 약 63MB | `runs/inserting_abl_d64_nosn_20261014T105353Z` |

`baseline_ema.pt`는 원본 `last.pt`(100만 스텝)에서 **EMA 가중치만** 꺼내 저장한 파일이다(`{"ema", "step", "config"}`). 평가와 추론 시점 제약 실험에는 이것만 있으면 된다. 같은 폴더의 `config.json`과 `normalizer.npz`가 함께 있어야 불러올 수 있으므로 폴더 단위로 다룬다.

`config.json`의 `hardflow_root`, `data_dir`는 원래 학습한 워크스페이스의 경로지만, 그 경로가 없으면 스크립트가 5절의 규칙으로 찾으므로 고칠 필요가 없다. 학습을 이어서 하려면 옵티마이저 상태가 든 원본 `last.pt`가 필요하다.

## 7. 불러와서 쓰기

```python
import sys
sys.path.insert(0, "/root/d3il-work/yflow-d3il/fm_baseline")
import numpy as np
from eval_pushing import load_model, Policy

run = "/root/d3il-work/yflow-d3il/fm_baseline/results/pushing_baseline"
hardflow = "/root/d3il-work/HardFlow"
cfg, model, norm, apply_conditioning, Solver, step = load_model(run, "baseline_ema.pt", "ema", hardflow, "cuda:0")
policy = Policy(model, norm, cfg, apply_conditioning, Solver, ode_steps=10, device="cuda:0")

obs = np.zeros(cfg["obs_dim"], dtype=np.float32)
plan = policy.plan(obs)
actions = plan[:, :2]
```

`plan`은 원래 단위의 `(16, 2 + obs_dim)` 배열이다. `load_model`은 `config.json`을 읽어 모델 크기를 자동으로 맞춘다.

## 8. 제약을 붙이는 위치

제약 조건은 원래 단위(m, rad)로 정의하고, 모델 내부에 적용할 때는 `normalizer.npz`로 정규화 공간으로 변환한다.

| 적용 시점 | 위치 | 예 |
|---|---|---|
| ODE 적분 중 (매 스텝) | `Policy.plan`의 `solver(x, t_span=...)` 부분. HardFlow `ConditionedODESolver.forward` 루프와 같은 구조로 직접 적분하면서 각 스텝에 보정을 넣는다. t=0 관측 위치는 매 스텝 고정해야 한다. | 매 스텝 투영, 가이던스 |
| 생성 직후 | `Policy.plan`이 반환하기 전, 또는 `plan` 반환 직후 | 종단 위치 보정, 궤적 투영 |
| 실행 직전 | `eval_*.py` 메인 루프의 `des = des + a` 직전 | 행동 필터, 작업 공간 경계 |
| 학습 손실 | `train.py`의 `flow.loss(...)` 이후 | 평활화 정규화 등. 체크포인트를 새로 학습해야 한다 |

추론 시점 제약은 베이스라인 체크포인트를 그대로 쓰고, 평가 스크립트의 실행 방식(4절)은 바꾸지 않는다.

### 베이스라인 실패 유형 (제약이 겨냥할 수 있는 지점)

- Pushing 실패 11/120: 목표 근처까지 밀었지만 5cm 안에 못 넣음 7건, 목표 손 위치가 실제 손에서 24~30cm까지 벌어짐 3건.
- Inserting 실패 61/328: 손이 멈춤 34건, 목표–손 거리 10cm 초과 30건(미로 벽에 막힌 상태로 목표 위치만 이동하는 것으로 추정), 평균 거리 2cm 미만의 근소한 실패 16건.

평가 폴더의 `rollouts.npz`에 rollout마다 목표 손 위치(`*_des`), 실제 손 위치(`*_c`), 블록 상태(`*_boxes`), 실행 행동(`*_act`)이 저장되므로, 제약 위반 지표는 이 파일로 계산할 수 있다.

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
