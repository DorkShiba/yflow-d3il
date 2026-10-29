# FM Constraint 단계별 통합 구현 계획

## 순서 수정 완료 (2026-10-29)

제약 적용 순서를 **Projection → PC → VLSA**로 수정한다. Projection은 Predict 내부에서, PC는 Predict 결과에서, VLSA는 PC까지 끝난 최종 계획에서 독립적으로 활성화한다. 이번 수정의 단계별 작업과 완료 조건은 **7절**을 따른다.

아래 2026-10-24 완료 기록과 1~6절은 기존 **Projection → VLSA → PC** 구현의 이력이다. 현재 코드와 `FM_Constraint_Integral.md`는 7절의 새 순서를 따른다. 충돌하는 순서·중복성·실패 처리 규칙은 7절을 우선한다. 구 순서를 실행하는 호환 분기는 남기지 않았다.

R0~R5를 완료했다. **81개 테스트 통과**, 두 실제 checkpoint의 8개 조합·단일 Predict 스텝·sample 진입점 검사, 두 D3IL 태스크 × 8개 조합의 각각 10스텝·2회 계획 생성, 최종 활성 단계 행동의 실행과 지도·카메라 재생을 확인했다. 카메라 재생의 최대 실제 손 XY 오차는 두 태스크 모두 `6.5e−7 m` 이내다. 모델·normalizer·데이터는 변경하지 않았다. 전체 성공률 실험은 별도다.

비교 기준과 로그·검증 스크립트는 `fm_constraint/runs/order_projection_pc_vlsa/`, 추적 가능한 요약은 [`order_projection_pc_vlsa_validation.json`](../fm_constraint/results/order_projection_pc_vlsa_validation.json)에 저장했다. 같은 실행 환경에서 all-off는 baseline과 정확히 일치하고, PC 또는 VLSA가 off인 6개 조합도 변경 전과 배열·상태·호출 수가 일치한다. 2026-10-24 저장 배열과의 작은 수치 차이는 요약에 별도로 기록했으며 현재 환경의 변경 전후 비교와 구분했다.

## 기존 구현 진행 현황 (2026-10-24, 구 순서 기준)

| 단계 | 상태 | 확인 내용 |
|---|---|---|
| 0 | 완료 | 기존 59개 테스트 통과, 소스·결과·모델 hash와 두 모델 기준 계획 저장 |
| 1 | 완료 | schema v2·중첩 override·세 방법 CLI·출력 경로 보호 |
| 2 | 완료 | 세 제약 행렬, 가속도·물체 margin 학습 통계 및 고정 YAML 값 |
| 3 | 완료 | OSQP 1.1.3 공통 QP, 해석해·독립 SLSQP·infeasible 검사 |
| 4~5 | 완료 | 통합 Predict/Projection/VLSA/PC, 조건·호출 예산·all-off 동등성 |
| 6 | 완료 | 두 실제 checkpoint의 8개 조합과 오프라인 sample 진입점 확인 |
| 7 | 완료 | 모의 환경 경계 검사, 실제 두 태스크×8조합 각각 10스텝·재계획 통과 |
| 8 | 완료 | 집계·지도/카메라 PNG/GIF 이관, 이전 결과 15개 보존·hash 검증. 실제 두 태스크 카메라 재생 오차 확인 |
| 9 | 완료 | 두 구 패키지가 없는 별도 복사본에서 실제 모델 sample 확인 후 구 폴더·이전 vlsa.py 제거 |

구 구현 전용 테스트를 새 제약 테스트로 교체한 최종 상태에서 **55개 테스트가 통과**했다. 두 실제 checkpoint의 all-off 계획은 통합 전 고정 seed 결과와 배열 단위로 정확히 일치한다. 추가로 baseline 학습 위임을 2 update로 확인했다. 가속도는 Pushing 13.7, Inserting 14.2 m/s²이며 물체 margin은 각각 0.25, 0.14m다. 공통 0.10m가 일부 초기 목표를 제외하는 문제를 확인한 뒤 **학습 초기 위치만으로** 고정 margin을 선정했다. 세부 산정 근거는 `fm_constraint/results/*_calibration.json`에 기록했다. 실제 D3IL은 `MUJOCO_GL=disable`로 확인했으며 이 검사는 전체 성공률 실험을 대신하지 않는다.

검사 산출물은 `fm_constraint/runs/integration_reference/`, 추적 가능한 검증 요약은 `fm_constraint/results/integration_validation.json`에 남긴다. 아래 구현 단계 설명은 작업 순서와 최종 완료 조건을 보존한다.

## 1. 목표와 적용 범위

[`FM_Constraint_Integral.md`](FM_Constraint_Integral.md)의 수학적 방침을 `fm_constraint/`에 구현한다. 최종 실행 패키지는 `fm_baseline/`과 `fm_constraint/`만 남기고, 검증과 결과 이관을 마친 뒤 `fm_projection/`, `fm_pc/`를 삭제할 수 있게 한다. 이 문서는 **구현 계획과 진행 기록**이며, 실제 완료 여부는 위 진행 현황과 검증 근거를 기준으로 한다.

기존 구현에서 확정했던 생성 흐름은 다음과 같다. 새 목표 흐름은 7.1절을 따른다.

```text
조건부 noise
  → Predict Euler [Projection 선택]
  → terminal 계획 QP [VLSA 선택]
  → 모델 재평가 + CBF-QP [PC Correct 선택]
  → 최종 검사
  → 앞 K개 실행 → 재계획
```

통합할 제약은 세 가지다.

| 제약 | 구현할 내용 |
|---|---|
| Possible Boundary | 명시한 사각형 workspace의 선형 부등식 |
| Acceleration Norm | 목표 증분 차분을 `Δt²`로 나눈 가속도; L2 한계의 내접 다각형 근사 |
| Object Envelope Tracking | 모든 블록 중심을 포함하는 볼록 정n각형; `n_sides`는 config에서 지정 |

태스크별 수치는 독립적으로 설정하되, 제약의 정의·QP 구성·샘플링 구현은 공유한다. 실제 손 대신 목표 명령에 부과하는 제약이라는 범위도 유지한다.

코드 작성 원칙은 다음과 같다.

- baseline과 같은 `train.py`, `sample.py`, `eval_pushing.py`, `eval_inserting.py` 진입점을 유지한다.
- 동일한 학습·모델 로딩·정규화·관측 구성은 baseline에서 import한다. 공통 코드를 복사해 두 벌로 관리하지 않는다.
- 새로운 알고리즘 코드는 평평한 소수 파일에 둔다. 여러 단계의 factory·registry·hook·adapter 계층을 만들지 않는다.
- 새로 작성하거나 의미를 바꾸는 핵심 함수의 docstring과 알고리즘 주석은 한글로 쓴다.
- 기존 baseline 가중치·데이터·학습 방법은 통합 때문에 변경하지 않는다.
- 이전 두 패키지는 이관 중에만 참고한다. 통합 코드가 이들을 import하도록 연결하지 않는다.

통합 방침 문서 9절의 세분화된 디렉터리는 **논리적 책임 구분**으로 해석한다. 실제 파일 배치는 아래의 간결한 구조를 적용한다. 기존 패키지의 영구 보존 대신 Git 이력·재현 메타데이터·결과 파일을 보존하는 방향으로 전환한다.

## 2. 최종 파일 구조

```text
configs/
  config.py                  # 기존 설정 로더 확장, schema 검증
  pushing.yaml               # Pushing의 모델·추론·제약 설정
  inserting.yaml             # Inserting의 모델·추론·제약 설정

fm_constraint/
  __init__.py                # 기존 baseline import·checkpoint 호환 처리
  train.py                   # baseline 학습 위임
  sample.py                  # baseline 데이터 도구 + 통합 정책 오프라인 평가
  eval_pushing.py            # Pushing 평가 진입점
  eval_inserting.py          # Inserting 평가 진입점
  policy.py                  # Predict → Projection → VLSA → PC 전체 흐름
  constraints.py             # 세 제약의 G,d 구성과 물리 단위 위반 검사
  qp.py                      # 공통 선형 부등식 QP 계산·잔차 검사
  evaluation.py              # 공통 실행 루프·이전 행동·큐·로그
  check_constraints.py       # 제약 수치 검사·학습 가속도 통계
  compute_metrics.py         # 공통 결과 집계·비교
  render_rollouts.py         # 공통 결과 재생·시각화
  tests/
    test_config.py
    test_constraints.py
    test_policy.py
    test_evaluation.py
  README.md
  results/legacy/            # 이관한 기존 결과와 재현 메타데이터
  runs/                      # 새로운 실행 결과; Git 추적 제외
```

기존 `vlsa.py`의 역할은 공통 제약과 QP로 흡수한다. 이전 구현을 참조하는 호출과 테스트가 모두 교체되면 파일을 제거한다. VLSA라는 방법 이름은 유지하되, 파일을 하나 더 유지하기 위해 같은 QP를 중복 구현하지 않는다.

`sampling/`, `constraints/` 하위 패키지나 태스크별 policy 클래스는 초기 통합에서 만들지 않는다. 하나의 파일에 독립적인 책임이 과도하게 늘어나는 경우에만 이후 분리를 검토한다.

### 2.1. 핵심 파일의 책임

| 파일 | 주요 책임 | 두지 않을 책임 |
|---|---|---|
| `policy.py` | 조건부 noise, Euler, 단계별 보정, 정규화 왕복, 모델 호출 계측 | 환경 실행, 데이터 split, 제약별 기하 수식 |
| `constraints.py` | 관측 해석, 사각형·가속도·정n각형의 면 구성, 장벽 scale, 위반 측정 | 모델 평가, YAML 파싱, QP 반복 알고리즘 |
| `qp.py` | `min 0.5‖z−z_nom‖² s.t. Gz≤d`와 solver 결과 검증 | 태스크 이름 분기, 모델·시뮬레이터 import |
| `evaluation.py` | 관측→계획→행동 실행, 재계획, 실패·성과 기록 | 제약식 재작성, 별도 샘플러 구현 |

Projection, terminal VLSA, PC는 `policy.py`에서 구분된 코드 블록으로 읽을 수 있게 한다. PC 반복처럼 독립적으로 검증할 필요가 있는 긴 계산은 이름이 분명한 메서드로 분리하되, 한두 줄짜리 연산마다 helper를 만들지 않는다.

## 3. baseline에서 재사용할 부분

| 통합 위치 | baseline에서 가져올 부분 | 통합에서 작성할 부분 |
|---|---|---|
| `train.py` | `fm_baseline.train.main()`과 기존 학습 구현 | 설정 전달·출력 경로 처리 |
| 모델 로딩 | `fm_baseline.eval_pushing.load_model()` | 통합 정책 객체 생성 |
| 정규화·데이터 | `fm_baseline.train.Normalizer`, `find_split_files`, `load_split`, `window_index`, 경로 해석 함수 | 직전 행동 선택, 제약 통계·보고 |
| Pushing 평가 | `setup_d3il`, `read_state`, `build_obs`, 관측 검사 | 공통 평가 루프 호출 |
| Inserting 평가 | `fm_baseline.eval_inserting`의 환경·관측·context 로딩 함수 | 공통 평가 루프 호출 |
| all-off 정책 | `fm_baseline.eval_pushing.Policy` | 활성 보정이 있는 경우의 통합 샘플링 |

현재 baseline 일부는 `from train import ...`, `from eval_pushing import ...` 같은 절대 import와 `sys.path` 조작을 사용한다. 이를 무시하고 여러 파일에서 import 우회 코드를 추가하지 않는다. 우선 기존 `fm_constraint.__init__.import_baseline()` 경로로 모듈을 읽고, 충돌 검사와 checkpoint의 좁은 호환 처리를 유지한다. 진입점은 `python -m fm_constraint.<module>`로 통일한다.

baseline `sample.py`의 주요 절차는 `main()` 안에 들어 있으므로 그대로 호출하면 통합 정책을 끼울 수 없다. 재사용 가능한 데이터·모델 로딩 함수는 import하고, 샘플 선택·통합 정책 호출·지표 계산 부분만 `fm_constraint/sample.py`에 작성한다. 함수 내부의 짧은 RMSE 식까지 재사용하려고 복잡한 추상화를 추가하지 않는다.

## 4. 단계별 구현 순서

각 단계는 아래의 **완료 조건**을 확인한 뒤 다음 단계로 진행한다. 다음 단계의 API를 맞추기 위한 최소 수정은 함께 할 수 있으나, 미구현 기능을 조용히 baseline으로 우회시키지 않는다.

### 단계 0. 현재 상태와 재현 기준 고정

**수정 대상:** 개발 기록과 결과 목록. 이 단계에서는 알고리즘을 변경하지 않는다.

작업:

1. baseline checkpoint·normalizer·모델 설정의 경로와 hash, 코드 커밋을 기록한다.
2. 이전 두 패키지의 실행 옵션, 결과 파일, 후처리·영상 도구, 외부 참조를 목록화한다. 추적되지 않은 결과도 포함한다.
3. 기존 `fm_constraint` 설정·학습·평가 테스트의 통과 여부를 기록한다. 사용 가능한 PyTorch·HardFlow·D3IL 환경도 확인한다.
4. 두 실제 checkpoint에 대해 고정 관측·seed의 baseline 계획을 재현 기준으로 확보한다. 실제 시뮬레이터 실행과 오프라인 샘플 확인을 구분한다.
5. 새 디렉터리 구조와 최종 삭제 방침을 통합 명세 문서에 동기화한다.

**완료 조건:** 무엇을 보존해야 하는지 목록이 있고, 이후 all-off 동등성을 비교할 기준이 있다. 이 단계에서 기존 폴더를 삭제하지 않는다.

### 단계 1. 설정 스키마와 실행 경로 정리

**수정 대상:** `configs/config.py`, 태스크 YAML, `train.py`, 진입점, `test_config.py`.

작업:

1. 명세의 `schema_version`, `control`, `sampling`, `methods`, `constraints`, `solver`를 설정 로더에 반영한다. CLI override도 중첩 키를 명시적으로 처리한다.
2. 세 방법의 on/off와 세 제약의 파라미터를 구분한다. 이전 실행 필터 전용 설정은 새 스키마에 암묵적으로 섞지 않는다.
3. `object_envelope.n_sides≥3`, 가속도 근사의 짝수 `polygon_sides≥4`, `dt_s>0`, 양의 가속도 한계·margin·장벽 scale 등을 검증한다.
4. `Projection final + terminal VLSA`의 중복 설정과 지원하지 않는 옵션은 오류로 처리한다.
5. 모델 구조·checkpoint 일치 검사, CLI 우선순위, 저장 경로 보호, `resolved_config.yaml` 저장을 유지한다.
6. 출력은 `fm_constraint/runs/`로 통일한다. `.gitignore`의 실제 경로도 확인하여 기존 `fm_constrain/runs/` 오타를 정리한다.
7. training-free 설정은 학습 loss에 전달하지 않는다. 학습 진입점에서 추론 보정이 켜진 설정은 명확히 오류로 처리한다.

가속도 값은 단계 2에서 선정한다. 그전의 `null`은 설정 템플릿 상태이며 활성 가속도의 검증을 통과시키지 않는다. baseline 회귀 검사는 별도의 명시적 제약-off 설정으로 진행할 수 있다. 구현이 아직 없는 활성 방법도 실행을 거부한다.

**완료 조건:** 설정 오류를 조기에 잡으며, 기존 가중치를 읽는 all-off 경로와 baseline 학습 위임이 유지된다. 이전 키를 새 의미로 조용히 해석하지 않는다.

### 단계 2. 세 제약의 기하·행렬·가속도 기준 구현

**수정 대상:** `constraints.py`, `check_constraints.py`, 태스크 YAML, `test_constraints.py`.

작업:

1. 현재 목표 `q₀`, 실제 손, 모든 블록, 직전 실제 실행 행동 `a_prev`, `dt`를 명시적으로 전달받는다. 관측 채널 순서는 baseline 규격을 따른다.
2. 행동 계획 `A=[a₀,…,a_(H−1)]`를 시간 우선 순서로 평탄화한다. 선택·누적·차분 행렬의 순서를 한 곳에 고정한다.
3. 사각형 workspace의 네 면을 각 horizon 시점에 구성한다.
4. 가속도는 `α_h=(a_h−a_(h−1))/Δt²`로 계산하고, 내접 정m각형의 선형 제약으로 변환한다. `h=0`에는 `a_prev`를 우변에 반영한다.
5. 물체 평균 중심·균등 법선·공통 내접 반경을 이용해 모든 블록을 포함하는 정n각형을 만든다. 이 기하는 한 계획 동안 고정한다.
6. `G,d`, 면의 제약 이름·시점, 장벽 scale을 함께 반환한다. 통합 명세와 같은 식을 사용하며 면마다 별도 클래스를 만들지 않는다.
7. QP 면 잔차, workspace·Object Envelope 위반, 정확한 L2 가속도 값과 초과량을 계산한다. m와 m/s²를 구분한다.
8. `check_constraints.py`에서 두 태스크의 **학습 시연 안에서만** 가속도 분포를 계산한다. 시연 경계를 넘는 차분을 만들지 않고, 시작 행동의 이전 값 0 조건도 별도 집계한다.
9. 분포·다각형에서의 시연 보존율을 근거로 태스크별 `max_m_s2` 초기값을 명시하고 선정 근거를 저장한다. validation으로 조정하더라도 test 성능으로 값을 선택하지 않는다.

**완료 조건:** 수식과 직접 계산한 위반값이 일치한다. 모든 블록이 Object Envelope에 포함되고, 가속도 다각형의 꼭짓점이 L2 원판 내부에 있다. 중복·공선 블록, 다른 n, 첫 시점·재계획 경계·단위 변환을 검사한다. 실제 실험 YAML에는 가속도 한계가 채워져 있다.

### 단계 3. 공통 QP 구현

**수정 대상:** `qp.py`, 필요한 의존성 선언, `test_constraints.py`.

작업:

1. 일반 `Gz≤d`와 양의 정부호 이차 목적을 처리하는 **한 가지 QP backend**를 선택한다. 현재 환경에서 지원하는 의존성과 infeasible·미수렴 상태 반환을 확인한 뒤 선정·버전을 기록한다. 여러 solver를 자동 선택하는 체계는 만들지 않는다.
2. 기본 목적함수는 raw 단위의 `0.5‖z−z_nom‖²`로 고정한다. 계획 행동 투영과 PC 벡터장 보정에 같은 함수를 사용한다.
3. NumPy `float64`를 기준으로 계산한다. solver status, 반복 수, 지연, 최대 물리 잔차를 반환한다.
4. XY축을 결합하는 다각형과 시점 간 가속도 차분을 지원한다. 기존 box 전용 `BoxQP`를 그대로 일반 solver로 간주하지 않는다.
5. 해 없음·미수렴·비유한값·허용 오차 초과를 실패로 반환한다. 경계를 clip해 문제를 바꾸거나 마지막 iterate를 정상 해로 사용하지 않는다.
6. 모델 dtype으로 변환한 후의 잔차 검사는 호출 측에서도 수행한다. QP 내부 정밀도만으로 최종 통과를 판정하지 않는다.

**완료 조건:** 해석해가 있는 작은 문제와 독립적인 수치 최적화 결과를 비교한다. 공동 제약이 feasible한 경우·infeasible한 경우, 내부점·경계점, dtype 왕복을 검증한다. 단순 순차 clipping과 다른 공동 최소 수정 결과를 확인한다.

### 단계 4. 통합 Policy의 Predict와 Projection 구현

**수정 대상:** `policy.py`, `test_policy.py`.

작업:

1. baseline의 로딩 결과와 `Policy`를 재사용하는 통합 정책을 만든다. 성공 시 raw `(H, action_dim+obs_dim)` 계획을 반환하는 인터페이스를 유지한다.
2. `plan(obs, previous_action, ...)`에서 직전 실행 행동을 명시적으로 받는다. episode 간 이 값을 정책 안에서 숨겨 관리하지 않는다.
3. 모든 방법 off인 일반 실행은 baseline `Policy.plan()`에 직접 위임한다. 추가 noise draw나 제약 기반 행동 거부가 없어야 한다.
4. 활성 방법이 있는 경로는 동일한 정규화·noise·현재 관측 조건으로 공통 Euler 생성 루프를 실행한다. 첫 관측 벡터장은 0이고 매 갱신 후 조건을 복원한다.
5. Projection on이면 Euler 갱신 직후 raw 행동을 공통 QP로 투영한다. `every`, `skip_first`, 비교용 `final`의 적용 시점을 명세대로 구현한다.
6. `predict_steps=1`도 지원한다. all-off 1스텝이 baseline solver의 최소 길이 가정에 걸리면 이 경우만 명시적 Euler 한 스텝을 사용하고, 식과 조건 처리를 별도로 검증한다.
7. 단계별 모델 호출 수, 제약 수정량·잔차를 `last_report` 등에 저장한다. 실패 시 계획 대신 `None`과 구조화된 실패 보고를 제공하여 평가 루프가 실행을 차단할 수 있게 한다.

동일 noise 비교를 위한 선택적 입력은 통합 경로에서 지원할 수 있다. 이를 위해 baseline의 RNG나 solver를 전역 monkeypatch하지 않는다. all-off 기준 비교는 같은 seed와 동일한 noise 생성 순서로 수행한다.

**완료 조건:** 다중 스텝 all-off가 baseline과 일치한다. 단일 스텝, 조건 유지, Projection 호출 수, 중간 raw 행동의 제약 만족, 추가 모델 호출이 없음을 검증한다.

### 단계 5. VLSA terminal 보정과 PC Correct 구현

**수정 대상:** `policy.py`, `test_policy.py`.

작업:

1. VLSA on이면 Predict 결과의 전체 행동 계획을 같은 QP로 한 번 보정한다. 모델을 호출하지 않으며 H개의 단일 행동을 순차 보정하는 방식으로 대체하지 않는다.
2. PC on이면 앞 단계 결과를 초기값으로 사용한다. 새 noise나 두 번째 Predict를 만들지 않고 보정 시간만 0부터 시작한다.
3. `α(1−τ)vθ`의 행동 성분을 raw 벡터장으로 변환하고, 공통 면으로 `h_j=(d_j−G_jA)/s_j`를 계산한다.
4. CBF 부등식 `G_jU≤s_j φ(h_j)`를 공통 QP로 푼다. 가속도 면도 같은 행렬에 포함한다.
5. 행동 벡터장만 교체하고 관측 성분은 명목 보정 벡터장으로 갱신한다. 첫 관측과 `a_prev`, 기하는 고정한다.
6. 최종 계획을 검사하되, PC 뒤에 숨은 투영을 추가하지 않는다. 실패는 그대로 보고한다.
7. 플래그 8개 조합을 지원한다. 기본 호출 수는 PC off 10회, on 19회이며, 동일 호출 예산 1+9·5+5는 명시적 별도 설정으로 다룬다.

**완료 조건:** VLSA와 `Projection final`의 동일성, feasible 계획에서의 VLSA 멱등성, `every` 뒤 VLSA의 중복성을 확인한다. PC는 조건 유지·가속도 면 반영·실패 전달·모델/QP 호출 수를 검사한다. 기존 PC와 제약·예측 횟수가 달라졌으므로 옛 출력과 같아야 한다는 테스트를 만들지 않는다.

### 단계 6. 오프라인 sample 통합

**수정 대상:** `sample.py`, `check_constraints.py`, 관련 정책 검사.

작업:

1. baseline의 데이터 로딩·정규화·window index를 import하고, 모델은 공통 `load_model()`로 읽는다.
2. 각 window `(episode,start)`에서 `start>0`이면 같은 시연의 `actions[start−1]`, 아니면 0을 이전 행동으로 사용한다.
3. baseline sampler를 직접 호출하던 부분을 통합 정책 호출로 교체한다. 최초 구현은 명확한 window 단위 실행으로 만들고, batching 최적화는 결과가 검증된 뒤 수행한다.
4. shape·유한값·첫 관측·행동/관측 RMSE와 함께 단계별 제약 위반·가속도·수정량·시간을 보고한다.
5. 실패 window 수와 전체 시도 수를 저장한다. 성공적으로 생성한 계획의 RMSE와 전체 실패율을 구분한다.
6. 입력 run 폴더에 출력하지 않고 `fm_constraint/runs/`에 설정과 검사 결과를 저장한다.

**완료 조건:** 실제 두 checkpoint로 8개 조합을 오프라인 실행할 수 있다. 동일 관측·noise를 이용한 비교와 이전 행동 연결이 정확하다. 환경 의존성 없이 제약·정책 검증을 실행할 수 있다.

### 단계 7. 공통 평가 루프에 연결

**수정 대상:** `evaluation.py`, 두 `eval_*.py`, `test_evaluation.py`.

작업:

1. 기존 환경·관측·context 함수를 baseline에서 import하고, 두 태스크가 같은 실행 루프와 정책 생성 경로를 사용하게 한다.
2. episode reset에서 `previous_executed_action=0`으로 시작하고, 매 실제 실행 후에만 갱신한다. 재계획이나 실패 시 임의로 초기화하지 않는다.
3. 계획 호출에 현재 관측과 이전 실행 행동을 전달한다. 반환 실패와 `last_report`를 확인한 뒤 큐를 구성한다.
4. `des ← des+executed_action`으로 목표 명령을 갱신하고 다음 관측에 반영한다. 생성 전체 H개 중 앞 K개만 실행한다.
5. 방법이 활성화된 경우 최종 검사 실패는 실행하지 않고 episode 실패로 기록한다. all-off의 같은 제약 위반은 측정만 한다.
6. 별도 `vlsa.execution_recheck` 옵션을 구현한다. 각 실행 직전 최신 기하·이전 실행 행동으로 H=1 QP를 풀고, 행동이 수정되면 실행 후 남은 큐를 폐기한다.
7. 실패한 시도와 실제 실행을 구분해 분모를 기록한다. 실제 접촉 정보가 없으면 접촉 기반 안전 성공률은 `null`을 유지한다.
8. reset·종료·예외 상황에서 이전 행동, 큐, 계획 ID, 로그가 다른 episode로 누출되지 않게 한다.

**완료 조건:** 모의 환경에서 실행 순서·큐·재계획·가속도 경계·실패 차단을 검증한다. 실제 D3IL에서 두 태스크의 all-off와 활성 조합을 짧게 실행한다. 실제 환경 검증을 수행하지 못하면 그 항목은 미완료로 남긴다.

### 단계 8. 결과 집계·시각화·실험 기록 이관

**수정 대상:** `compute_metrics.py`, `render_rollouts.py`, 평가 출력, `README.md`.

작업:

1. 두 기존 패키지의 유사한 후처리 도구에서 필요한 기능을 하나로 합친다. baseline에 같은 기능이 없다면 통합 패키지가 소유한다.
2. 새 실행 결과의 `results.json`, `rollouts.json`을 공통 형식으로 정한다. 이미 저장된 legacy NPZ는 별도 명시적 읽기 경로로 지원하거나 한 번 변환한다. 원본은 보존한다.
3. 제약별 잔차·정확한 가속도 값·단계별 개입·실패율·모델/QP 호출·시간을 집계한다. 기존의 행동 차분 지표를 단위 확인 없이 가속도나 jerk로 재명명하지 않는다.
4. context를 묶은 성공률·엔트로피·score 비교와 신뢰구간을 유지한다. 서로 다른 checkpoint·호출 예산·실행 묶음을 구분한다.
5. 시각화에는 사각형, 계획 시점의 물체 다각형, 실제 손·명령 궤적을 표시한다. 기록된 시점의 기하를 사용한다.
6. 기존 `results/` 자료를 `fm_constraint/results/legacy/{projection,pc}/`로 이관하고 원래 경로·커밋·설정·파일 hash를 목록에 남긴다.
7. 실행 명령·출력 형식·가속도 한계 선정 근거와 기본값을 README에 정리한다.

**완료 조건:** 새 결과와 이관한 이전 결과를 각각 읽고 집계·시각화할 수 있다. 기존 도구를 호출해야만 가능한 기능이 남아 있지 않다. 새 제약이 없는 과거 로그에서 새 안전 지표를 추정해 채우지 않는다.

### 단계 9. 전체 검증 후 기존 패키지 제거

**수정 대상:** 기존 두 패키지, 이전 `vlsa.py`와 불필요한 테스트, 관련 import·문서·실행 명령.

삭제 전 확인:

- [x] baseline 파일·가중치·데이터가 의도치 않게 변경되지 않았다.
- [x] 새 패키지의 일반 실행·sample·평가·후처리가 이전 두 패키지를 import하지 않는다.
- [x] 설정·제약·QP·정책·평가 테스트가 통과했다.
- [x] 두 실제 checkpoint에서 all-off 동등성과 8개 조합 오프라인 검사가 통과했다.
- [x] 두 실제 D3IL 태스크에서 8개 조합의 짧은 실행과 실패 처리를 확인했다. 실제 검증이 막혀 있으면 삭제 완료로 처리하지 않는다.
- [x] 이전 결과·설정·코드 커밋을 보존했고, 원본/이관 파일의 hash와 개수를 확인했다.
- [x] metric·영상 도구와 사용 중인 실행 명령을 새 패키지로 바꿨다.
- [x] 이전 폴더를 제외한 별도 checkout에서도 새 진입점 import·오프라인 검사가 동작한다.

조건 충족 후 `fm_projection/`, `fm_pc/`를 삭제한다. 구 경로의 호환 wrapper나 재수출 import를 남기지 않는다. 과거 알고리즘 재실행은 기록한 Git 커밋으로 checkout해서 수행하고, 새 실행에 legacy 구현을 다시 끼우지 않는다.

이전 `vlsa.py`와 기존 제약 전용 테스트도 참조가 사라졌는지 확인한 뒤 정리한다. 새 제약 테스트로 보장을 대체하지 않은 상태에서 테스트만 삭제하지 않는다. 마지막으로 README, 통합 명세, 이 계획의 완료 상태와 경로를 갱신한다. 문서의 역사 설명·재현 manifest에 있는 옛 이름은 남겨도 되지만 실행 의존성은 없어야 한다.

**완료 조건:** `fm_constraint`와 baseline 의존성만으로 모든 신규 기능이 동작하고, 기존 두 디렉터리를 실제로 제거한 상태에서 확인이 끝났다.

## 5. helper와 주석 작성 원칙

### 5.1. 함수 분리 기준

- 여러 곳에서 재사용하는 비자명한 계산이나, 수학적으로 독립 검증할 계산만 함수로 분리한다.
- 세 제약은 `constraints.py` 안의 이름이 분명한 코드 블록으로 구성한다. 면 하나·좌표 하나마다 함수를 만들지 않는다.
- 공통 QP 함수는 Projection·VLSA·PC가 공유한다. 세 방법별 solver wrapper를 중복 작성하지 않는다.
- 정책은 단계 순서가 눈에 보이는 `plan()`과 필요한 소수 메서드로 작성한다. 여러 no-op hook을 따라가야 흐름을 알 수 있는 구조는 피한다.
- config 읽기, 로깅, 단순 배열 변환을 위해 범용 `utils.py`를 먼저 만들지 않는다.
- 에러·결과 자료구조는 shape·단위·status를 명확히 설명한다. 동일 정보를 여러 계층의 객체로 감싸지 않는다.

### 5.2. 한글 주석에 포함할 내용

핵심 함수 docstring에는 입력 shape·단위, 반환값, 실패 의미를 쓴다. 알고리즘 주석에는 연산의 목적과 그 식이 필요한 이유를 쓴다. 단순 대입문을 그대로 읽어 주는 주석은 생략한다.

```python
# 첫 행동의 이전 값은 직전에 실제 실행한 행동이다.
# 재계획마다 0으로 바꾸면 계획 경계에서 가속도 제한이 끊어진다.
previous_action = previous_executed_action

# 행동은 목표 위치의 증분(m)이다. 따라서 인접 행동의 차이를
# 제어 간격의 제곱(s²)으로 나눠 목표 명령 가속도(m/s²)를 계산한다.
# 생성 시간의 Euler 보폭과 혼동하지 않도록 control_dt를 사용한다.
acceleration = (action - previous_action) / control_dt**2

# 모든 블록을 포함하는 다각형을 현재 관측에서 한 번 만든다.
# PC 반복 중에는 같은 면을 유지해야 고정된 장벽식이 성립한다.

# 내접 다각형을 사용하므로 cos(pi / m)을 곱한다.
# 이 항을 빼면 원 밖의 가속도를 허용하는 외접 근사가 된다.
```

추가로 다음 위치에는 이유를 설명하는 한글 주석이 필수다.

1. action-first 채널 순서와 raw↔normalized 변환; 벡터장에는 center를 더하지 않는 이유.
2. 첫 관측의 벡터장을 0으로 하고 매 갱신 뒤 조건을 재적용하는 위치.
3. 가속도 첫 행에 `a_prev`가 들어가는 우변과 PC에서 그 값이 상수인 이유.
4. Object Envelope에 성공한 물체도 포함하는 규칙과 margin의 바깥 방향.
5. QP status 외에 dtype 복귀 후 물리 잔차를 검사하는 이유.
6. VLSA terminal 단계와 실행 재검사의 차이, 실행 수정 후 큐를 폐기하는 이유.
7. all-off에서는 위반을 측정해도 행동을 수정·거부하지 않는 분기.

## 6. 단계별 확인 범위와 최종 완료 기준

테스트는 수식·경계 조건·실제 동작을 확인하는 데 집중한다. 함수 호출 여부만 복제하는 테스트, 설정의 모든 줄을 그대로 반복하는 테스트는 늘리지 않는다.

| 확인 수준 | 목적 | 시행 시점 |
|---|---|---|
| 독립 수치 검사 | 제약식·단위·QP 정답·불가능성·dtype 오차 | 단계 2~3 |
| 정책 검사 | 첫 관측, 이전 행동, 플래그 조합, 모델/QP 호출 수 | 단계 4~5 |
| 실제 checkpoint 오프라인 검사 | 실제 모델·정규화와 통합이 맞는지 확인 | 단계 6 |
| 모의 환경 평가 | 큐·재계획·실패·통계 분모 검증 | 단계 7 |
| 실제 D3IL 단기 실행 | 관측·명령·제어 간격·태스크 연결 검증 | 단계 7~9 |
| 전체 실험 | 성능·다양성·비용 비교 | 기능 검증 이후 별도 수행 |

긴 학습이나 전체 성능 실험을 단순 리팩터링 검증에 반복 사용하지 않는다. 반대로 실제 환경 검증을 오프라인 통과만으로 대체하지 않는다. 단계별 변경에 필요한 검사를 수행하고, 새 변경이나 실패 원인이 없으면 같은 검사를 불필요하게 반복하지 않는다.

최종 완료 상태는 **통합 policy·sample·eval·후처리가 하나의 패키지에서 동작하고, 세 제약과 세 방법의 조합이 검증되며, 이전 두 패키지를 제거해도 실행 가능함**이다. 성능 향상 여부는 별도 실험 결과로 판단한다.

## 7. Projection → PC → VLSA 순서 수정

### 7.1. 수정 목표와 고정할 의미

```text
조건부 noise
  → Predict Euler [Projection 선택: every 또는 비교용 final]
  → 모델 재평가 + CBF-QP [PC Correct 선택]
  → terminal 계획 QP [VLSA 선택]
  → 최종 검사
  → 앞 K개 실행 → 재계획
```

세 플래그는 서로를 자동으로 켜거나 끄지 않는다. PC는 VLSA의 선행 투영 없이 Predict 결과를 직접 받는다. VLSA는 PC가 켜져 있으면 PC 결과를, 꺼져 있으면 Predict 결과를 받으며, 전체 H개 행동을 한 번에 공동 보정한다. 실행 직전 H=1 재검사는 기존의 별도 옵션으로 유지하고 기본 비교에서는 끈다.

세 방법은 한 계획에서 같은 `PlanConstraints`의 `G,d`, 장벽 scale, 물체 기하, `a_prev`, `dt_s`를 사용한다. 순서 수정으로 제약 정의·QP 목적·모델·정규화·학습 데이터·태스크별 한계값을 바꾸지 않는다. 새 noise나 두 번째 Predict를 추가하지 않는다.

아래 표의 모든 경로 뒤에는 같은 최종 검사가 붙는다. 기본 `projection.mode=every`, `execution_recheck=false` 기준이다.

| Projection | PC | VLSA | 목표 경로 |
|---|---|---|---|
| off | off | off | Euler → 실행 |
| on | off | off | 투영 Euler → 실행 |
| off | on | off | Euler → PC → 실행 |
| off | off | on | Euler → terminal QP → 실행 |
| on | on | off | 투영 Euler → PC → 실행 |
| on | off | on | 투영 Euler → terminal QP → 실행 |
| off | on | on | Euler → PC → terminal QP → 실행 |
| on | on | on | 투영 Euler → PC → terminal QP → 실행 |

### 7.2. 수정 단계 R0 — 변경 전 기준과 영향 범위 고정

**대상:** `policy.py`, 설정 검증, 정책·평가 테스트, `sample.py`, 집계·시각화, 기존 검증 자료의 조사와 새 기준 저장.

작업:

1. 현재 커밋·작업 트리 차이, 두 checkpoint·normalizer hash, resolved config, 실행 환경을 기록한다. 기존 변경은 보존한다.
2. 고정 관측·이전 행동·noise로 두 태스크의 8개 조합을 저장한다. 최종 계획뿐 아니라 `last_report`의 단계별 행동·위반량·상태·모델/QP 호출 수도 확보한다.
3. 현재 `policy.py`의 VLSA 선행 블록, `configs/config.py`의 `Projection final + VLSA` 무조건 금지, `test_policy.py`의 플래그 순서와 멱등성 가정, 문서의 구 순서를 수정 대상으로 명시한다.
4. `integration_reference/`, `integration_validation.json`, legacy 결과는 구 순서의 근거로 보존한다. 새 산출물은 예를 들어 `runs/order_projection_pc_vlsa/`, `results/order_projection_pc_vlsa_validation.json`처럼 별도 경로에 저장한다.

**완료 조건:** 새 결과와 비교할 수 있는 구 순서의 재현 기준과 영향 파일 목록이 있다. 기존 55개 테스트 통과 기록을 새 순서의 검증으로 재사용하지 않는다.

### 7.3. 수정 단계 R1 — 명세·설정·로그의 순서 의미 정리

**대상:** `FM_Constraint_Integral.md`, `configs/config.py`, 태스크 YAML의 설명, `test_config.py`, `README.md`, 결과 메타데이터.

작업:

1. 설계 문서의 방법 역할·전체 흐름·기하 고정 구간·6절 보정 설명·7절 조합표·8절 중복 규칙·9절 API·12절 검사·13절 구현 순서를 새 순서에 맞춘다. 특히 `terminal_correct`를 `pc_correct` 다음으로 이동한다.
2. 기존 `methods.*` 키·CLI·수치 기본값은 유지한다. `terminal_plan`은 **활성 PC까지 끝난 계획**을 뜻하도록 설명하고, 순서 선택 옵션이나 구 순서 실행 분기를 새로 만들지 않는다.
3. `Projection final + VLSA`는 **PC off일 때만** 같은 입력에 대한 연속 투영이므로 기존 중복 금지를 유지한다. PC on이면 두 투영 사이에 PC가 있으므로 허용한다. 해당 설정의 허용·거부를 각각 검사한다.
4. `Projection final` 단독과 VLSA 단독의 동일성, `Projection every → VLSA`의 멱등성은 **PC off 조건**으로 한정한다. PC가 개입한 조합의 VLSA를 무조건 중복으로 간주하지 않는다.
5. `last_report`, 오프라인 sample, 평가 결과와 검증 요약에 고정 순서를 식별할 메타데이터를 남긴다. 예: `pipeline_order: [projection, pc, vlsa]`. 활성 플래그와 별도로 저장하여 같은 schema v2 설정이라도 구 실행과 구별한다. 이 필드가 없는 과거 결과는 새 순서로 추정하지 않고 커밋·manifest를 참고하거나 순서 미상으로 표시한다.

**완료 조건:** 문서·설정 검증·로그가 같은 순서를 설명한다. 설정 키 나열 순서나 CLI 인자 순서가 실행 순서를 결정하지 않는다.

### 7.4. 수정 단계 R2 — Policy를 Predict → PC → VLSA로 변경

**대상:** `fm_constraint/policy.py`와 직접 관련된 정책 검사.

작업:

1. `plan()`의 기존 VLSA 블록을 PC 블록 뒤·최종 검사 앞으로 이동한다. 공통 `_project()`를 재사용하고, PC 수식과 QP backend는 유지한다. 모듈 docstring·한글 단계 주석·최종 검사 주석도 수정한다.
2. PC는 Predict가 반환한 동일 `x`에서 시작한다. 첫 관측 조건을 유지하며, 모델·행동 벡터장의 raw 변환과 고정 기하·이전 행동 규칙을 유지한다.
3. 성공한 활성 단계의 행동·위반량은 `predict → pc → vlsa` 순서로 기록한다. 비활성 단계의 결과를 복제하여 실행한 것처럼 기록하지 않는다. VLSA 수정량은 PC on이면 PC 종료 행동에 대한 값으로 해석한다.
4. PC QP의 infeasible·미수렴, 비유한 출력 등으로 유효한 계획이 없으면 즉시 실패한다. 뒤의 VLSA가 solver 실패를 덮거나 마지막 iterate를 되살리지 않는다. Predict/Projection 실패 역시 뒤 단계로 넘기지 않는다.
5. PC가 정상적으로 유한 계획을 반환했지만 행동 잔차가 허용 오차를 넘는 경우, PC 단계 잔차를 로그에 남긴다. VLSA on이면 명시적인 terminal QP를 적용한 뒤 최종 판정하고, off이면 최종 검사에서 실패시킨다. PC의 유한 반복만으로 초기 infeasible 계획의 복구를 보장하지 않는다.
6. VLSA 자체의 QP 실패·dtype 복귀 후 잔차 실패는 `failure_stage=vlsa`로 전달한다. 이후 추가 보정 없이 반환 dtype의 최종 행동을 검사한다. VLSA off일 때 숨은 terminal 투영을 붙이지 않는다.

**완료 조건:** PC와 VLSA를 함께 켜면 VLSA가 실제 PC 출력 전체를 받는다. PC 단독의 실패 판정과 VLSA를 명시적으로 켠 최종 보정을 구분하며, 실패 계획은 실행하지 않는다.

### 7.5. 수정 단계 R3 — 순서를 구별하는 회귀 검사

**대상:** `test_policy.py`, `test_config.py`, 필요한 `test_constraints.py`·`test_evaluation.py` 검사.

작업:

1. 조합 테스트의 표기 순서를 `(projection, pc, vlsa)`로 통일하거나 이름으로 참조한다. 기존 `flags[1]`, `flags[2]`를 그대로 사용하여 호출 수 기대값을 뒤바꾸지 않는다.
2. 고정 noise와 입력 계획에 따라 출력이 달라지는 모델을 사용해 **PC와 투영의 순서를 바꾸면 결과가 달라지는 수치 사례**를 만든다. 공통 제약·QP로 계산한 `VLSA(PC(Predict(...)))`와 실제 결과를 비교하고, 구 순서로 되돌리면 실패하는지 확인한다. 호출 목록만 검사하는 테스트로 대체하지 않는다.
3. 중간 행동 기록으로 PC 입력이 Predict 종료값이고 VLSA 입력이 PC 종료값임을 확인한다. 관측 채널 유지·동일 제약 기하·이전 행동·가속도 면·dtype 복귀 검사를 함께 유지한다.
4. PC의 정상 종료 후 잔차를 VLSA가 수정하는 사례와, 같은 조건에서 VLSA off이면 최종 위반으로 실패하는 사례를 검사한다. PC solver 실패 시 VLSA 미실행, VLSA 실패 시 실행 차단도 검사한다.
5. all-off의 baseline 배열·RNG 동등성을 유지한다. 같은 입력에서 **PC 또는 VLSA 중 하나라도 off인 6개 조합**은 구 순서와 같은 결과·상태·호출 수여야 한다. 두 방법을 함께 켠 2개 조합은 새 순서의 수치 기준으로 판정한다.
6. PC off에서만 기존 투영 동일성·멱등성을 요구한다. PC on에서 VLSA가 수정할 수 있음을 검사하고, 항상 수정해야 한다는 기대는 두지 않는다.
7. 정상 종료 기준 모델 호출 수는 `N_p + I_pc N_c`, QP 호출 수는 `Q_projection + I_pc N_c + I_vlsa`다. `every`의 `Q_projection=max(N_p-skip_first,0)`, `final`의 값은 1이며 Projection off이면 0이다. 기본 10/19회와 별도 1+9·5+5 예산을 확인하고, 실패 시에는 실제 수행한 호출까지만 센다. 실행 재검사 QP는 별도로 집계한다.
8. 전체 설정·제약·정책·평가 테스트를 실행한다. 실행 재검사 off에서 최종 VLSA 결과가 큐에 들어가는지, on에서 실제 수정 행동으로 이전 행동을 갱신하고 큐를 폐기하는지도 확인한다.

**완료 조건:** 단순 블록 이동 실수·숨은 선행 VLSA·중복성 오판을 잡는 검사가 통과한다. 새 테스트 통과 수와 명령을 새 검증 요약에 기록한다.

### 7.6. 수정 단계 R4 — 두 실제 모델·평가·후처리 확인

**대상:** 두 태스크의 `sample`·`eval` 진입점, `evaluation.py`, `compute_metrics.py`, `render_rollouts.py`, 새 검증 자료.

작업:

1. 두 실제 checkpoint에서 같은 관측·이전 행동·seed 또는 noise로 8개 조합의 오프라인 검사를 수행한다. R0의 6개 불변 조합은 배열·상태·호출 수를 비교하고, PC+VLSA 2개 조합은 단계 순서·terminal 잔차·실패 전달을 확인한다. `predict_steps=1`도 별도 확인한다.
2. `sample.py`와 공통 평가 루프가 새 Policy 결과와 단계 로그를 그대로 사용하는지 확인한다. 집계·시각화에 단계 순서 가정이 있으면 수정하고, 과거 로그의 읽기 호환성을 유지한다.
3. 동일 플래그여도 구 순서와 새 순서의 결과가 구분되도록 집계 표·그림의 메타데이터 또는 라벨에 순서를 표시한다. 순서가 다른 비교는 변경 효과 비교라고 명시하고 하나의 동일 방법 결과로 합치지 않는다.
4. 실제 D3IL에서 두 태스크 × 8개 조합을 짧게 실행한다. 기존 기준처럼 최대 10스텝·`execute_steps=8`로 재계획 경계를 포함하고, 실행된 행동이 terminal 결과에서 왔는지와 실패 계획이 실행되지 않는지 확인한다. 재현 가능한 실패도 숨기지 않고 단계·원인을 기록하며, 예상하지 못한 실패는 수정 후 재검사한다.
5. 각 태스크의 새 로그로 집계·지도 및 카메라 재생 산출물을 확인한다. 기준 경로·명령·환경·hash·단계 잔차·호출 수·실패 사례를 새 검증 요약에 저장한다. 실제 환경 검증이 불가능하면 해당 항목을 미완료로 둔다.

**완료 조건:** 실제 모델과 두 태스크의 실행·재계획·실패 처리·후처리까지 새 순서로 확인했다. 짧은 실행 결과를 전체 성공률 검증으로 해석하지 않는다.

### 7.7. 수정 단계 R5 — 문서 완료 기록과 실험 기준 갱신

**대상:** 이 문서의 수정 진행 상태, `FM_Constraint_Integral.md`, `fm_constraint/README.md`, 새 검증 요약과 실행 예시.

작업:

1. 순서·8개 조합·중복 설정 규칙·PC 종료 잔차 처리·호출 예산을 문서와 코드에서 대조한다. 기존 1~6절은 구 순서의 이력으로 유지하고, 새 사용자가 따라야 할 기준은 7절과 갱신된 명세임을 명시한다.
2. R0~R4의 근거 경로와 완료 여부를 기록한다. 구 `integration_validation.json`을 새 순서 통과 기록으로 덮어쓰지 않는다.
3. 본 실험은 새 순서와 동일 checkpoint·context·seed·제약값·호출 예산으로 별도 수행한다. PC+VLSA 두 조합의 과거 성능을 새 순서의 성능으로 재사용하지 않는다. 이번 수정 완료 조건은 기능 검증까지이며 전체 성능 실험은 별도 기록한다.

수정 진행 체크리스트:

- [x] R0: 구 순서의 재현 기준과 영향 범위 확보 (`runs/order_projection_pc_vlsa/before/`)
- [x] R1: 명세·설정·로그 의미 동기화 (`pipeline_order`와 PC 조건부 중복 검증)
- [x] R2: Policy를 Projection → PC → VLSA로 변경 (구 순서 실행 분기 없음)
- [x] R3: 수치 순서 검사·회귀 검사 통과 (81개 테스트)
- [x] R4: 두 실제 모델·D3IL·후처리 확인 (`after/`, `d3il_validation.json` 및 요약의 산출물 경로)
- [x] R5: 완료 근거·사용 문서·실험 기준 갱신 (`results/order_projection_pc_vlsa_validation.json`)

**최종 완료 조건:** 독립적인 세 플래그가 고정된 **Projection → PC → VLSA** 순서로 동작하고, 최종 검사·실행 큐·로그가 그 결과를 사용하며, 구 순서 결과와 구별되는 검증 근거가 남아 있다.
