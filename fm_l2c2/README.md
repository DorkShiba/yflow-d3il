# Method 4: L2C2 기반 Flow Matching 정책 정규화

`feature/method4`의 학습 시점 방법이다. 기본 CFM 손실에 국소 정책 민감도 손실을 추가한다. 기존 학습 엔진·데이터 전처리·모델·EMA·평가 코드를 재사용하며, 평가 때 별도의 필터를 추가하지 않는다. 실제 D3IL 성공률 개선은 아직 검증하지 않았다.

## 원 논문과 적용 범위

Kobayashi, **L2C2: Locally Lipschitz Continuous Constraint towards Stable and Smooth Reinforcement Learning**, IROS 2022. 구현 확인 원문은 [arXiv v1](https://arxiv.org/pdf/2202.07152v1)의 식 (12), (14), (15), (16)이다.

원 논문은 actor-critic 강화학습에서 정책의 **squared Hellinger distance**와 가치함수 차이를 정규화한다. 여기서는 모방학습이고 정책 밀도를 직접 계산하지 않으므로, 가치함수 항을 사용하지 않고 **동일한 생성 노이즈를 공유하는 두 행동 청크의 제곱 거리**를 사용한다. 따라서 원 논문을 그대로 재현한 것이 아니라 **L2C2의 국소 이웃·거리 가중을 채택한 FM 변형**이다. 이 손실이 충돌 회피나 Lipschitz 상한을 보장하지 않는다.

정규화된 시연 윈도우의 첫 두 관측을 `s`, `s_next`로 사용한다. 기존 로더가 시연 경계를 넘지 않는 윈도우만 생성하므로 서로 다른 에피소드의 상태를 연결하지 않는다.

```text
half_width = sigma + (sigma - 1) * epsilon
u_j ~ Uniform(-half_width, half_width), independently per coordinate
s_neighbor = s + (s_next - s) * u
d = max_j(abs(u_j) for coordinates with s_next_j != s_j) + epsilon

A  = policy_theta(s,          z)[:8, :2]
A' = policy_theta(s_neighbor, z)[:8, :2]  # exactly the same noise z
action_distance = 0.5 * mean_over_8_steps(sum_over_XY((A - A')**2))
L_l2c2 = mean_over_batch(action_distance / d)
L_total = L_CFM + weight * L_l2c2
```

- 변화가 없는 상태 차원은 이웃에서도 고정하고 상대 거리 기여를 0으로 두어 0/0을 피한다. 모든 차원이 정지한 전이는 같은 정책 출력으로 손실 0이 된다.
- `A`는 **정규화 공간**의 목표 손 XY 증분이다. 실제 단위의 가속도·jerk나 인접 행동 차이 자체를 손실로 쓰는 것은 아니다.
- `sin(yaw), cos(yaw)`도 원문의 좌표별 상자에서 독립적으로 교란한다. 이웃은 단위원을 벗어날 수 있는 합성 관측이며 실제 시뮬레이터 상태로 간주하지 않는다. 이웃을 clipping하거나 재정규화하지 않는다.
- 정책 출력은 베이스라인과 동일한 Euler 10회로 생성한다. 원 solver의 `detach`만 제거한 미분 가능한 구현을 사용하고 초기 관측을 고정한다. 미래 관측은 모델이 생성하며 정답 미래 관측을 정책에 주입하지 않는다.
- 두 정책 분기 모두 미분한다. 임의의 벡터장 한 번만 비교하는 대체 손실은 쓰지 않는다.
- 작은 부분 배치에서 계산해 메모리를 제한한다. 정규화용 난수는 학습 step으로 결정하고 전역 난수 상태를 복원하므로 기본 FM 노이즈·데이터 추출 순서를 바꾸지 않는다.
- `weight=0`이면 정책 추가 호출과 난수 생성 없이 기본 CFM 학습이 유지된다.

## 파일 및 기존 코드 변경

| 파일 | 역할 |
|---|---|
| `regularizer.py` | 이웃 샘플링, 미분 가능한 정책 생성, 거리 가중 손실 |
| `train.py` | 전용 CLI와 공통 학습 엔진용 확장 객체 |
| `tests/` | 이웃 정의·역전파·난수 보존·off 동일성·학습 재개·추론 호환성 검증 |

`fm_baseline/train.py`에는 선택적인 학습 확장 인터페이스와 추가 로그 열만 넣었다. 기존 CLI는 그대로 동작한다. L2C2 전용 진입점에서만 확장이 켜진다. 기존 평가·체크포인트 형식을 유지한다.

## 학습 실행

기존 README에 따라 HardFlow 지정 커밋과 D3IL, Python 환경을 준비한 뒤 **저장소 루트**에서 실행한다. 새로운 의존성은 없다. 학습은 처음부터 진행하며 기본 베이스라인과 동일하게 Adam, WD=0, SN 없음, seed 42, 배치 32, lr 2e-4, 단일 cosine, EMA 0.995, 100만 업데이트를 사용한다.

```bash
# 먼저 짧은 실행으로 메모리, 학습 속도와 두 손실의 비율을 확인
python -m fm_l2c2.train --task pushing --device cuda:0 \
  --steps 1000 --log-every 100 --save-every 500 --tag l2c2_pilot

# Pushing: dim 32, dim_mults 1,4,8 자동 선택
python -m fm_l2c2.train --task pushing --device cuda:0 --tag l2c2_w001

# Inserting: dim 64, dim_mults 1,2,4,8 자동 선택
python -m fm_l2c2.train --task inserting --device cuda:0 --tag l2c2_w001

# 동일 학습 엔진의 off 대조군
python -m fm_l2c2.train --task pushing --device cuda:0 \
  --l2c2-weight 0 --tag l2c2_off

# 저장된 설정·모델·Adam·EMA·난수 상태를 복원
python -m fm_l2c2.train --resume codex/l2c2/runs/<run> --device cuda:0
```

짧은 pilot은 짧은 cosine 주기의 별도 실행이다. 이를 100만 업데이트 학습으로 연장하는 용도로 쓰지 않는다. 중단/재개 시에는 처음 계획한 총 step 수를 유지한다. `--resume`은 학습 실행 폴더의 `last.pt`를 요구하며, 베이스라인 `baseline_ema.pt`는 optimizer가 없는 평가용 파일이라 재개에 사용할 수 없다. 재개 시 현재 CLI보다 저장된 학습 설정을 우선한다. 기본 학습 진입점으로 L2C2 실행을 재개하려 하면 오류를 낸다.

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--l2c2-weight` | 0.01 | 정규화 계수. 검증된 최적값이 아닌 초기 실험값 |
| `--l2c2-sigma` | 1 | 실제 상태 전이 대비 이웃 크기 |
| `--l2c2-epsilon` | 0.1 | 상대 거리 분모의 안정화 값 |
| `--l2c2-batch-size` | 4 | 매 학습 배치 중 정규화에 사용할 최대 윈도우 수 |
| `--l2c2-ode-steps` | 10 | 정규화 정책 생성 횟수. 본 비교에서는 추론과 같게 유지 |
| `--l2c2-action-steps` | 8 | 비교하는 실행 예정 행동 수. 본 비교에서는 execute-steps와 같게 유지 |

전체 배치 32에 기본 부분 배치 4라면 별도로 두 분기 × 4윈도우 × 10회 모델 계산과 그 역전파가 추가된다. GPU 메모리나 시간은 실측해야 한다. 작은 계수 후보를 **validation으로** 선택하고 test에서 선택하지 않는다. `sigma`, `epsilon`, 배치 수 등을 함께 바꾸면 별도 실험으로 기록한다.

출력은 기본 `codex/l2c2/runs/` 아래에 저장되고 Git에서 제외된다. 위치는 `--out-root`로 지정할 수 있다. `config.json`에는 변형의 정의·하이퍼파라미터를 기록한다. `metrics.csv`의 `train_loss_mean`은 총 손실이고 `fm_loss`, `l2c2_loss`, `weighted_l2c2_loss`, `action_distance`, `relative_distance` 열이 추가된다. `val_loss`와 `val_loss_ema`는 비교 가능한 기존 CFM 검증 손실이다.

## 평가

```bash
# Pushing: baseline과 같은 30 context x 4회, 10 ODE steps, 8 actions 실행
python fm_baseline/eval_pushing.py --run-dir codex/l2c2/runs/<run> \
  --ckpt last.pt --weights ema --n-contexts 30 --n-traj 4 \
  --ode-steps 10 --execute-steps 8 --seed 0 --tag l2c2

# Inserting: baseline과 같은 82 context x 2회 x 2 seeds
python fm_baseline/eval_inserting.py --run-dir codex/l2c2/runs/<run> \
  --ckpt last.pt --weights ema --n-traj 2 --seed 0 \
  --ode-steps 10 --execute-steps 8 --tag l2c2_s0
python fm_baseline/eval_inserting.py --run-dir codex/l2c2/runs/<run> \
  --ckpt last.pt --weights ema --n-traj 2 --seed 1 \
  --ode-steps 10 --execute-steps 8 --tag l2c2_s1
```

성공률 외 엔트로피·D3IL 점수·행동 변화량·jerk를 함께 비교한다. `fm_pc/compute_metrics.py`는 기존 평가 형식의 rollout 지표 계산에 재사용할 수 있다. 초기 학습 seed 하나에서 좋게 나왔다고 일반적인 향상으로 결론 내리지 않는다. Methods 1/2/3과의 조합은 먼저 이 체크포인트를 해당 평가기에 전달해 추론 시 제약을 추가하는 방식이며, 통합 32조합 실행기는 이번 구현에 포함하지 않았다.

## 테스트

```bash
HARDFLOW_ROOT=/root/d3il-work/HardFlow python -m pytest fm_l2c2/tests \
  -q -p no:cacheprovider --basetemp codex/l2c2/test_tmp
```

`pytest`는 개발 검증 전용이다. `HARDFLOW_ROOT`가 없으면 실제 모델 학습 통합 테스트는 skip된다. 로컬 PyTorch 2.6 이상에서 팀의 기존 체크포인트 로더를 시험할 때는 신뢰하는 자체 생성 체크포인트에 한해 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`이 필요할 수 있다. 팀 실행 환경은 기존 requirements의 PyTorch 2.3.1을 사용한다.
