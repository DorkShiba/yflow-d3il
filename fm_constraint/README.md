# FM Constraint

기존 `fm_baseline/`을 수정하지 않고 YAML 설정과 실행 안전 필터 실험을 추가하는 패키지다. VLSA/VSLA/DLSA 표기 중 코드·설정 키는 `vlsa`로 통일한다. 여기의 원형 CBF-QP는 AEGIS 전체 재현이 아니다.

## 현재 구현 범위

| PLAN 단계 | 상태 |
|---|---|
| 1. 설정·import 경계 | 구현 및 테스트 완료 |
| 2. 기존 학습·sampling 재사용 | 구현. 두 실제 baseline EMA의 CPU sampling 및 짧은 학습·재개 확인 |
| 3. 필터 off 평가 | 공통 루프·진입점 구현. 모의 환경에서 기존 실행 순서 대조; 실제 D3IL 동등성 검증은 대기 |
| 4. 제어·기하·내부 step 감사 | 원본 커밋의 초기·목표·테이블 형상 확인. 로컬 D3IL과 gym이 없어 실제 운동·접촉 감사는 대기 |
| 5. QP | 목표 XY 직사각형 울타리 구현·평가 연결. 원형 장애물 CBF는 수치 검증만 완료 |
| 6. 실제 장애물·접촉 기반 비교 | 명령·제어 시점 실제 손의 울타리 위반 지표 구현. 실제 장애물·내부 step 접촉 비교는 대기 |

기본 제약은 물리 벽을 추가하지 않는 **목표 TCP 좌표의 가상 직사각형 울타리**다. D3IL 의존성이 준비되면 `--vlsa`로 실행할 수 있다. 실제 외부 장애물을 지정하는 원형 CBF 실험은 제어·기하 감사 전까지 중단한다. 현재 환경에서는 실제 시뮬레이터 검증을 완료하지 않았다.

**데이터 패딩 및 종단 데이터 보완은 구현 금지다.** 기존 구간 추출을 그대로 import한다. 마지막 행동은 학습되지만 시연 마지막 15개 상태는 첫 관측 조건이 되지 않고, 마지막 원본 상태는 현재 로더가 읽지 않는 `next_obs[-1]`에만 있다. 종단 재계획·유지에 미치는 영향은 알려진 미검증 한계로 남긴다.

## 설정과 실행

저장소 루트에서 실행한다. 기존 `requirements.txt`의 의존성을 사용하며 YAML은 PyYAML로 읽는다. 테스트는 pytest, 독립 QP 교차 검증은 SciPy를 사용한다. 수치 안전 필터 자체는 NumPy만 사용한다. 실제 평가에는 별도 D3IL 설치와 해당 의존성이 필요하다.

```bash
python -m fm_constraint.sample --config configs/pushing.yaml --set sample.n_windows=2
python -m fm_constraint.sample --config configs/inserting.yaml --set sample.n_windows=2
python -m fm_constraint.train --config configs/pushing.yaml --device cuda:0
python -m fm_constraint.eval_pushing --config configs/pushing.yaml --no-vlsa
python -m fm_constraint.eval_inserting --config configs/inserting.yaml --no-vlsa
python -m fm_constraint.eval_pushing --config configs/pushing.yaml --vlsa
python -m fm_constraint.eval_inserting --config configs/inserting.yaml --vlsa
python -m fm_constraint.eval_inserting --config configs/inserting.yaml --check-obs 5
python -m fm_constraint.eval_inserting --config configs/inserting.yaml \
  --export-contexts fm_constraint/runs/contexts/inserting_val.pkl
python -m pytest fm_constraint/tests -q
```

기본 YAML은 이식 가능한 확인을 위해 `device: cpu`다. 본 학습 기본값은 **100만 업데이트**이므로 짧은 검사는 `--set train.steps=2 --set train.batch_size=2 --set train.val_batches=1` 등으로 명시한다. `paths.hardflow_root`, `paths.d3il_root`, `paths.raw_root`는 실제 설치 위치에 맞춘다. Pushing 관측 점검의 `raw_root`는 `all_data` 폴더를 가리켜야 한다.

설정 우선순위는 명시한 CLI > YAML > 코드의 태스크별 기본값이다. 일반 override는 `--set SECTION.KEY=VALUE`를 반복 사용한다. 값은 YAML 타입으로 해석한다. 알 수 없는 키, 문자열 `"false"`를 boolean에 전달하는 경우, checkpoint 구조 불일치는 오류다. 상대 경로는 YAML 파일 위치·현재 작업 디렉터리가 아닌 저장소 루트 기준이다.

```python
from fm_constraint.config import ExperimentConfig

cfg = ExperimentConfig.from_yaml(
    'configs/pushing.yaml',
    overrides={'device': 'cpu', 'eval.execute_steps': 8},
)
cfg.validate('eval')
print(cfg.vlsa_enabled)
print(cfg.values['model'])
```

`--vlsa`와 `--no-vlsa`는 `vlsa.enabled`보다 우선한다. 기본 `geometry: rectangle`은 workspace 네 경계, `margin_m`, `command_delta_limit_m`를 사용한다. Pushing은 `[0.20, 0.85] × [-0.40, 0.50]m`, Inserting은 `[0.00, 0.85] × [-0.40, 0.75]m`이며 margin은 0.01m, 축별 명령 증분 한계는 0.01m다. 이 범위는 원본 형상·학습 데이터 포함 여부를 확인해 정한 연구용 영역이며 원본의 강제 경계나 로봇 도달 가능성 보장이 아니다. 구체적인 근거·범위는 `docs/DLSA_d3il_exp.md`의 0절에 있다.

직사각형은 `low <= 현재 목표 XY + 행동 <= high`를 QP에 직접 부과하므로 실제 속도 모델·gamma·푸셔 반경이 필요하지 않다. 체크인된 YAML의 null 값은 후속 원형 CBF 전용 파라미터다. 원형 수치 검증은 `geometry: circle`, `workspace: null`, gamma·반경·장애물을 지정하며, 실제 장애물 평가는 아직 차단한다. train/sample에서는 VLSA on을 오류로 처리한다.

모델의 dim, dim_mults, attention, horizon, SN은 **새 학습**에 설정할 수 있다. 기존 가중치 평가에서는 원래 `config.json`과 같은 구조여야 하며 관측·행동 차원, 순서도 고정한다. 재개는 기존 저장 args와 일치하는 설정만 허용하고, `train.resume`은 새 패키지 소유 run만 가리킬 수 있다.

## 재사용과 출력

- `train.py`: `fm_baseline.train.main()` 호출. 데이터 로딩·정규화·구간 추출·CFM·EMA·checkpoint 구현을 복사하지 않는다.
- `sample.py`: `fm_baseline.sample.main()` 호출. 원래 RMSE·조건 유지 검사를 그대로 실행한다.
- 평가: 기존 `load_model`, `Policy`, 태스크별 관측·환경·context 함수를 import하고 실행 루프만 공유한다. 실행 직전 `vlsa.enabled`를 확인해 `vlsa_constraint()`를 직접 호출하며, off에서는 VLSA 계산을 건너뛴다. 정지 비교군은 같은 VLSA 위험 판정을 별도로 사용한다.
- 기존 코드의 절대 import 이름 충돌은 오류로 처리한다. `python -m` 진입점을 사용하고 동일 프로세스에서 다른 `train` 모듈을 먼저 import하지 않는다.
- PyTorch 2.6 이후의 weights-only loading에는 기존 checkpoint의 `TorchVersion` 메타데이터만 좁은 context로 허용한다. `torch.load`를 monkeypatch하거나 weights-only 제한을 전체 해제하지 않는다.

출력·resume은 symlink를 해석한 `fm_constraint/runs/` 아래로 제한한다. baseline sampling은 `sample.no_save=true`여야 한다. train은 별도 부모 폴더 아래 기존 trainer가 run을 생성한다. 실제 생성 run과 부모에 `resolved_config.yaml`을 저장한다. 평가는 모델 설정 사본·checkpoint SHA-256·결과·step 로그를 새 폴더에 저장한다. 기존 입력 경로에 쓰지 않는다.

평가에서 물리 접촉 계측이 없으면 Safe Success Rate와 금지 접촉 없는 비율은 `null`이다. 울타리는 on/off 모두 `command_workspace_violation_rate`(목표 위반 에피소드 비율), `measured_workspace_violation_rate`(제어 시점 실제 손 위반 비율), `command_workspace_success_rate`(목표 영역을 유지한 작업 성공 비율)를 별도로 기록한다. 초기 reset과 각 제어 step을 검사하므로 내부 미세 step 전체의 안전 지표는 아니다. 로그는 `results.json`, `rollouts.json`이며 step별 경계 여유와 실패 원인을 포함한다.

## 수치 안전 필터

`vlsa.py`의 `vlsa_constraint(nominal_delta, measured_xy, command_xy, config)`가 VLSA 계산을 담당한다. `config`는 YAML의 `vlsa` 항목이다. 직사각형에서는 다음 목표 위치의 네 경계와 축별 증분 한계 아래 `0.5||delta-delta_nom||²`를 최소화한다. 원형 수치 모드에서는 `h=||p-c||²-R²`의 선형 CBF 제약을 사용한다. 둘 다 활성 제약 0·1·2개 후보를 열거하는 동일 2차원 solver로 풀며, 사각형의 독립 clipping 해·원형의 반공간 해석해·SciPy 최적화와 비교한다.

원형 모드의 기본 수치 모델은 `p_dot=delta/dt`이며, 선택적인 `tracking_gain=k`는 `p_dot=k*(q+delta-p)`다. 두 모델 모두 실제 D3IL 제어기에서 측정·검증되지 않았다. 직사각형은 이 근사를 사용하지 않으며 **목표 좌표만 제한하고 실제 손·로봇 링크·블록·내부 장벽 안전을 보장하지 않는다.** 결과는 action, h, residual, intervention_norm, latency_ms와 상태를 포함한다. invalid_input, initial_unsafe, infeasible은 action=None으로 반환하며 원래 위험 행동으로 fallback하지 않는다.

평가 루프는 VLSA가 켜져 있으면 큐의 **각 행동** 실행 직전에 최신 실제 위치와 현재 목표 위치로 `vlsa_constraint()`를 호출한다. 수정된 목표 위치는 다음 정책 관측에 반영한다. VLSA는 FM 학습 loss나 sampling 내부를 변경하지 않는다. 이 연결은 모의 환경 테스트로 검증했다. 실제 장애물 등록·geometry 쌍·내부 step 접촉 계측·추종 모델 감사가 다음 작업이다.
