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

저장소 루트에서 HardFlow·D3IL과 Python 환경을 준비한다. 기본 설정은 CPU다.

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
