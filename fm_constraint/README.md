# FM Constraint

기존 `fm_baseline`의 모델·정규화·학습·태스크 관측 코드를 import하여 세 training-free 방법을 통합한다. 설정은 schema v2이며, 과거 실행 필터의 YAML을 묵시적으로 변환하지 않는다.

```text
noise → Euler [Projection 선택] → terminal QP [VLSA 선택]
      → 모델 + CBF-QP [PC 선택] → 검사 → 앞 K개 실행 → 재계획
```

## 구조

| 파일 | 역할 |
|---|---|
| `policy.py` | 조건부 생성과 세 방법의 순서·on/off·비용·실패 관리 |
| `constraints.py` | 사각형·가속도·모든 블록을 포함하는 정n각형의 공통 행렬 |
| `qp.py` | OSQP를 이용한 `min 0.5‖z−z_nom‖², Gz≤d` |
| `sample.py` | baseline 데이터 도구와 통합 정책을 이용한 오프라인 검사 |
| `evaluation.py`, `eval_*.py` | 이전 실행 행동·큐·재계획을 관리하는 공통 평가 |
| `check_constraints.py` | 수치 검사·학습 데이터 기반 설정값 선정 |
| `compute_metrics.py`, `render_rollouts.py` | 결과 비교와 저장 궤적 지도·카메라 PNG/GIF |
| `train.py` | baseline trainer 위임; 추론 보정은 학습에서 비활성 |

공통 QP는 현재 환경에서 검증한 `osqp==1.1.3` 하나를 사용한다. Python API와 상태 반환은 [OSQP 공식 문서](https://osqp.org/docs/interfaces/python.html)를 따른다. solver 미수렴·해 없음·물리 잔차 초과에서는 실행할 행동을 반환하지 않는다.

## 제약과 기본 수치

세 방법은 같은 제약·기하를 사용하고 행동 채널만 수정한다. 현재 관측은 매 스텝 고정한다.

| 제약 | 정의 | Pushing | Inserting |
|---|---|---|---|
| workspace | 누적 목표 XY가 margin 적용 사각형 안에 위치 | `[0.21,0.84] × [−0.39,0.49] m` | `[0.01,0.84] × [−0.39,0.74] m` |
| 가속도 | `‖a_h−a_(h−1)‖₂ / dt² ≤ α_max` | `13.7 m/s²` | `14.2 m/s²` |
| 물체 영역 | 모든 블록 중심을 포함하는 정n각형 | `n=8`, margin `0.25 m` | `n=8`, margin `0.14 m` |

가속도는 **목표 명령의 이산 가속도**이며 실제 로봇 가속도와 구분한다. 제어 간격은 `0.035 s`다. L2 한계는 32면 내접 다각형으로 보수적으로 근사한다. Object Envelope의 면수 `n_sides`와 가속도 근사의 `polygon_sides`는 별도 설정이다.

가속도 한계는 시연 시작의 이전 행동 0도 포함하여 모든 학습 차분을 허용하는 다각형 한계의 1.05배를 올림한 값이다. 물체 margin은 학습 시연의 모든 초기 목표를 포함하는 최소 필요 margin에 1cm 여유를 주고 올림했다. 근거는 `results/*_calibration.json`이다. 초안의 공통 `0.10 m`는 Pushing의 일부 초기 위치를 제외하여, 고정 가속도 제한과 공동 적용하면 시작부터 infeasible이었다. 현재 margin은 실행 중 자동 확대하지 않는다.

workspace는 연구용 명령 영역이며 로봇 도달 가능성 증명이 아니다. 내부 장벽·블록 충돌·전체 링크 안전을 강제하지 않는다. 접촉 관측이 없으면 `safe_success_rate`는 `null`이다.

## 실행

저장소 루트에서 Python 환경과 HardFlow·D3IL 경로를 맞춘다. 상대 경로는 저장소 기준이며 출력은 `fm_constraint/runs/`에만 쓴다.

```bash
# 모든 방법 off: 기존 baseline과 같은 정책
python -m fm_constraint.sample --config configs/pushing.yaml --set sample.n_windows=4

# 세 방법을 독립적으로 선택; 모든 조합 지원
python -m fm_constraint.sample --config configs/pushing.yaml --projection --vlsa --pc \
  --set sample.n_windows=4
python -m fm_constraint.sample --config configs/inserting.yaml --pc --set sample.n_windows=4

# 시뮬레이터 없는 수치·학습 통계 검사
python -m fm_constraint.check_constraints --config configs/pushing.yaml --n-obs 4
python -m fm_constraint.check_constraints --config configs/pushing.yaml --calibrate-acceleration
python -m fm_constraint.check_constraints --config configs/pushing.yaml --calibrate-envelope

# 상태 기반 headless 평가: GPU/디스플레이 없이 MuJoCo의 GL을 비활성화 가능
MUJOCO_GL=disable python -m fm_constraint.eval_pushing --config configs/pushing.yaml --projection --vlsa --pc
MUJOCO_GL=disable python -m fm_constraint.eval_inserting --config configs/inserting.yaml --pc

# 중첩 키 변경과 개별 방법 해제
python -m fm_constraint.sample --config configs/pushing.yaml --vlsa --no-pc \
  --set constraints.object_envelope.n_sides=12 --set sample.n_windows=4

# 기존 학습 위임: 추론 방법은 모두 off여야 함
python -m fm_constraint.train --config configs/pushing.yaml
python -m pytest fm_constraint/tests -q
```

CLI > YAML > 기본값 순서다. `--no-projection`, `--no-vlsa`, `--no-pc`는 해당 YAML flag보다 우선한다. 수치 통계 명령은 설정을 자동 덮어쓰지 않는다. 면수·법선 각도를 바꾼 실험은 변경된 기하의 타당성·시연 보존율을 다시 확인한다.

기본 Predict는 10회이며 PC를 켜면 Correct 9회를 더해 모델 19회다. 동일 호출 예산 비교는 `--set sampling.predict_steps=1 --set methods.pc.correct_steps=9`처럼 명시한다. 생성 1회에서는 Projection도 생성 중 한 번만 적용된다.

VLSA는 현재 terminal **전체 계획**의 공동 QP다. `Projection final`과 동일한 최적화이므로 중복 설정을 금지한다. `Projection every`의 마지막 투영 뒤 VLSA는 같은 기하에서 중복 보정일 수 있다. PC 뒤에는 숨은 추가 투영이 없다.

## 실행 경계와 출력

직전 **실제 실행 행동**을 다음 계획의 첫 가속도 제약에 연결한다. episode 시작에서만 0으로 초기화한다. 기본은 전체 16개 생성 후 앞 8개 실행이다.

`--set methods.vlsa.execution_recheck=true`를 명시하면 VLSA on일 때 각 실행 직전 최신 관측으로 H=1 QP를 다시 푼다. 실제 행동이 수정되면 남은 큐를 폐기하고 재계획한다. 기본값은 false다.

- `results.json`: episode별 결과, 성공률·엔트로피·score, 실패율, 모델/QP 호출 수.
- `rollouts.json`: 실제 실행 행동·이전 행동·최신 물체 기하와 계획별 중간 행동·QP 상태.
- `resolved_config.yaml`, `model_config.json`: 설정과 checkpoint 구조. 모델·normalizer hash도 결과에 기록한다.
- sample의 `samples.npz`: 생성 성공 window와 정답·이전 행동. 실패율은 전체 시도 기준, RMSE는 생성 성공 window 기준이다.

```bash
python -m fm_constraint.compute_metrics --eval-dir fm_constraint/runs/<baseline> \
  --eval-dir fm_constraint/runs/<experiment> --label baseline --label combined
python -m fm_constraint.render_rollouts --eval-dir fm_constraint/runs/<evaluation> --episode c000_t0 --gif
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m fm_constraint.render_rollouts \
  --eval-dir fm_constraint/runs/<evaluation> --camera --gif
```

집계는 context 단위 bootstrap을 사용한다. checkpoint·task·context가 일치할 때만 paired 차이를 계산한다. 이전 summary-only 결과는 기존 집계값만 표시하고 새 지표나 신뢰구간을 만들어 채우지 않는다. 지도 재생은 저장된 명령·실제 손·당시 기하를 사용한다. `--camera`는 동일 context에서 실제 실행 행동만 MuJoCo에 재입력하고 재생 오차를 JSON으로 저장한다. 두 모드 모두 모델을 다시 실행하지 않는다.

## 이관 상태

기존 결과 15개는 `results/legacy/`에 보존하고 SHA-256을 검증했다. 두 태스크의 실제 8개 조합·카메라 재생·구 패키지가 없는 별도 복사본의 오프라인 실행을 확인한 뒤 `fm_projection/`, `fm_pc/`와 이전 `vlsa.py`를 제거했다. baseline 소스·가중치·데이터는 변경하지 않았다. 이전 소스는 `results/integration_source_manifest.json`의 Git 커밋으로 재현할 수 있다. 단계별 근거는 `docs/PLAN.md`, 추적 가능한 검증 요약은 `results/integration_validation.json`에 기록한다.
