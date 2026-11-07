# FM Constraint 단계별 통합 구현 계획

## 모델 코드 독립화 (2026-11-07)

현재 실행은 `fm_baseline/models/`의 모델·CFM·조건부 Euler를 사용한다. 외부 HardFlow 의존성을 제거했으며 원본 출처는 `THIRD_PARTY_NOTICES.md`, 검증 근거는 `fm_constraint/results/local_models_validation.json`에 기록한다. 아래 이전 구현·실험 기록의 HardFlow 경로는 이력이다. 자세한 내용은 **9절**을 따른다.

## PF2MP 교체 구현 완료 (2026-11-06, 기능 검증)

이번 변경의 기준은 **8절**이다. 기존 PC Correct를 주석으로 보존하고 실행 경로에서 제외한 뒤, PF2MP를 Predict의 벡터장 보정으로 넣는다. 새 흐름은 **Predict 내부 `[FM + PF2MP → Euler → Projection]` → terminal VLSA**이며, Projection·PF2MP·VLSA는 독립적으로 on/off한다. schema v3 코드·설정·KDE 캐시를 구현하고 기능 검증을 완료했다. 근거와 범위는 **8.9절** 및 `fm_constraint/results/pf2mp_validation.json`을 따른다. 전체 성공률·충돌률 본실험은 별도다.

아래 7절과 [`FM_Constraint_Integral.md`](FM_Constraint_Integral.md)의 이전 명세는 Projection → PC → VLSA의 이력이다. 새 순서·PC 비활성화·중복 투영·호출 예산·최종 검사 규칙은 8절을 우선한다. 관련 명세와 README에도 현재 기준을 반영했다. 과거 완료 기록은 그대로 보존한다.

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

## 8. PC → PF2MP 교체 구현 계획 (2026-11-06)

### 8.1. 목표와 변경 범위

사용자 요청에 따라 현재 PC Correct의 반복문·CBF-QP 계산 블록을 **줄 주석으로 보존하고 실행하지 않도록** 변경할 계획이다. 복구용 커밋도 기록한다. PF2MP는 이 위치에 새 Correct 루프를 만드는 것이 아니라 **공통 Predict 루프의 벡터장 계산에 통합**한다. 원리는 [`PF2MP.md`](PF2MP.md), 기존 명시 제약·QP·실행 큐의 기준은 [`FM_Constraint_Integral.md`](FM_Constraint_Integral.md)를 따른다.

```text
PF2MP on일 때: 추론 프로세스 초기화 → train KDE 캐시 검증 / 없으면 생성 → 한 번 로드

현재 관측으로 조건부 noise 생성
  → Predict N회:
      현재 X에서 FM 벡터장 계산
      현재 X에서 에너지 gradient 계산 [PF2MP 선택]
      두 벡터장을 합하여 Euler 적분
      적분 결과의 행동 계획을 QP로 투영 [Projection 선택]
      현재 관측 조건 복원
  → 전체 행동 계획의 terminal QP [VLSA 선택]
  → 유한값 검사·제약 측정·선택한 강제 정책에 따른 판정
  → 앞 K개 행동 실행 → 최신 관측으로 재계획
```

기존 모델·EMA checkpoint·normalizer·학습 손실·세 명시 제약의 수식과 물리 단위는 유지한다. PC의 추가 모델 호출과 별도 보정 시간 초기화는 제거한다. 이번 작업은 추론 변경이며 재학습은 요구하지 않는다.

### 8.2. PF2MP와 Projection의 결합 수식

생성 시간을 $\tau$, 제어 간격을 $\Delta t=0.035$초로 구분한다. 정규화된 계획 $X_k=[\widetilde A_k,\widetilde S_k]$에서 PF2MP는 **상태의 위치 채널 벡터장**, Projection은 **Euler가 만든 중간 샘플의 행동 계획**을 수정하므로 함께 적용할 수 있다.

$$
Y_{k+1}=X_k+\frac1N\left[v_\theta(X_k,k/N)+\lambda G_{\rm PF}(X_k)\right],
\qquad G_{\rm PF}=[0_{\rm action},\,M_s\nabla_{\widetilde S}\Psi].
$$

$M_s$는 선택한 상태 위치 채널과 고정되지 않은 계획 시점만 통과시키는 마스크다. **Action 증분 채널의 PF guidance는 항상 0**이며 명목 FM 행동 벡터장은 그대로 적분한다. Projection/VLSA가 활성화되면 그 단계에서는 행동이 수정될 수 있다.

Projection on이면 $Y_{k+1}$를 raw 단위로 복원한 행동 $A_Y$에 대해 기존 공통 QP를 푼다.

$$
A^*=\arg\min_A\frac12\|A-A_Y\|_2^2
\quad\text{s.t.}\quad GA\le d.
$$

여기서 QP 행렬 $G$는 PF guidance 텐서 $G_{\rm PF}$와 별개인 기존 명세의 기호다. 구현에서는 `guidance`와 `constraints.G`로 구분한다. 수정한 행동만 재정규화하여 결합하고 첫 관측을 복원한다. Projection off이면 Euler 결과를 그대로 쓴다.

PF와 FM은 **동일한 갱신 전 $X_k$** 에서 평가하며, Projection은 갱신 후 샘플에 정확히 한 번 적용한다. PF 안에서 QP·clipping을 수행하거나 Projection 안에서 KDE 에너지를 다시 최소화하지 않는다. 마지막 PF 보정을 Projection 뒤에 덧붙이지 않는다. 현재 단계의 PF 직접 가산과 행동 투영은 서로 다른 채널에 작용하지만, 다음 FM 평가는 전체 계획을 입력받으므로 반복 과정에서 두 방법은 간접적으로 상호작용한다.

### 8.3. D3IL에 사용할 에너지와 좌표계

첫 구현은 **train 목표 TCP XY의 2D Gaussian KDE**를 사용한다. 행동 증분의 밀도는 장벽의 공간 위치를 표현하기 어려우므로 주 실험의 기본으로 삼지 않는다. 실제 TCP 위치 KDE, 상대 좌표 KDE, 관측 조건부 KDE는 별도 확장으로 남긴다.

**데이터 집계 원칙:** 같은 태스크라면 블록 초기 배치·행동 경로·처리 순서가 달라도 전체 train 시연을 하나의 모집단으로 모아 공통 PDF를 추정한다. 현재 context와 비슷한 시연만 골라 매번 별도 PDF를 만들지 않는다. Pushing과 Inserting은 태스크별로 분리하고 val/test는 밀도 fitting에서 제외한다. 전체 train에서 support를 부분 추출하는 것은 계산 근사이며, 모든 점을 반드시 kernel 중심으로 써야 한다는 뜻은 아니다. 이 밀도는 여러 상황에서 방문한 위치의 주변분포로, 현재 블록 배치에서의 조건부 안전 확률은 아니다.

**채널 적용 원칙(사용자 지정):** PDF 입력은 위치이며 action 증분이 아니다. PF는 생성된 **미래 상태의 목표 TCP XY**에만 적용한다. `[action(2), obs]` 배열 기준 대상은 `X[:,1:,2:4]`이고, `X[:,:,:2]`의 guidance는 엄밀히 0으로 둔다. 실제 TCP XY·블록 위치·sin/cos 채널도 기본 guidance는 0이다. 첫 시점의 전체 관측은 조건으로 고정한다. 여기서 0으로 두는 것은 action 자체나 FM 행동 벡터장이 아니라 **에너지의 action 방향 미분과 PF 가산항**이다.

생성 관측에서 읽은 raw 목표 위치를 $q_h^{\rm state}$라 두고 다음 에너지를 사용한다. Action 누적에서 위치를 재구성하여 미분하지 않는다.

$$
\phi(q)=\log\widehat p(q)+\alpha d(q,\mathcal H),\quad
\alpha<0,\quad \mathcal H=\{q:\log\widehat p(q)>\ell_\eta\}.
$$

$$
\Psi(S)=\sum_{h=1}^{H-1}\phi(q_h^{\rm state}),\qquad
E=-\Psi,\qquad v_{\rm guided}=v_\theta-\lambda\nabla E
=v_\theta+\lambda\nabla\Psi.
$$

`energy`를 비용 $E$로 정의하면 음의 gradient, 논문의 potential $\Psi$를 쓰면 양의 gradient를 적용한다. 코드·설정·로그에서 이 규약을 명시한다. Sum 규약을 고정하고, horizon을 바꾸는 비교에서는 유효한 guidance 크기도 확인한다.

상태의 목표 위치에 해당하는 center·대각 scale을 $c_q,S_q$라 하면 $q_h^{\rm state}=c_q+S_q\widetilde q_h$이므로 다음을 사용한다.

$$
\nabla_{\widetilde q_h}\Psi=S_q^\top\nabla_q\phi(q_h^{\rm state})\quad(h\ge1),
\qquad \nabla_{\widetilde A}E=0.
$$

Gaussian의 해석적 score와 거리항의 gradient에 위치 scale을 곱하고 유한차분으로 대조한다. Guidance 텐서를 0으로 초기화한 뒤 허용된 위치 slice에만 값을 채운다. Autograd를 사용해도 FM 모델이나 행동 누적 계산을 에너지 그래프에 연결하지 않는다. `guidance[:,:,:act_dim] == 0`과 `guidance[:,0,act_dim:] == 0`을 확인한다.

이 규칙은 앞서 제안했던 **누적 목표 위치의 에너지를 action으로 역전파하는 설계를 대체**한다. `PF2MP.md`의 행동 chain rule 및 의사 코드도 본 절의 상태 위치 전용 방식으로 교체했다. 이번 구성은 상태·행동을 함께 생성하는 D3IL 모델에 맞춘 적용안으로 구분한다.

**효과와 한계:** PF가 수정한 미래 상태를 다음 Euler 단계의 FM이 입력으로 읽으며 행동에 간접 영향을 줄 수 있지만, 동일 단계에서 action을 직접 수정하지는 않는다. 따라서 마지막 단계의 PF는 이후 모델 평가가 없어 행동으로 전달되지 않으며, `predict_steps=1`이면 이 경로를 통한 행동 변화가 없다. 생성된 목표 위치와 $q_0+\sum_{j\le h}a_j$의 차이를 별도로 기록하고, 상태 에너지 개선만으로 실행 행동의 안전 개선을 판정하지 않는다. Projection/VLSA는 계속 행동 누적으로 제약을 계산한다.

KDE는 raw XY(m)로 정의하여 checkpoint의 normalizer와 독립시킨다. Bandwidth 단위는 m이며 log-density도 같은 좌표 규약으로 비교한다. Gaussian 혼합의 score는 `logsumexp`/softmax로 안정화한다. 초기 noise가 분포에서 멀어도 정의되는 해석적 KDE를 기본으로 사용한다. 저밀도에서 PDF가 수치적으로 0이 된 것을 밀도장이 없는 것으로 처리하지 않는다.

거리항은 논문에 구현 상세가 없으므로 초기 구현에서는 **학습 데이터 범위와 명시적 padding을 덮는 2D grid로 level set을 이산화하고, 고밀도 grid 점 집합까지의 최근접 거리**로 근사한다. Grid 원점·간격·범위·threshold 규칙을 고정하고, 고밀도 영역이 grid 외곽에서 잘리지 않는지 확인한다. 임의 query에서 집합까지의 거리를 계산하므로 grid 밖 query도 처리할 수 있다. 연속 level set까지의 거리와 다른 근사임을 기록하고, grid 의존성·최근접점 전환·거리 0에서의 gradient 규약을 검증한다. 고밀도 점이 없으면 설정 오류로 처리한다. 밀도와 score 자체는 grid 보간으로 대체하지 않고 Gaussian KDE로 평가한다.

Bandwidth, $\alpha$, $\ell_\eta$, $\lambda$는 논문에서 D3IL 확정값으로 가져올 수 없다. 학습 분포와 밀도 그림으로 후보를 정하고 조정용 validation에서 선택하여 고정한다. Threshold를 train log-density의 분위수로 정하면 대상점·분위수·실현값을 기록한다. 거리항이 없는 설정은 명시적 ablation으로 구분한다.

### 8.4. 데이터셋 재확인

계획 작성 시 실제 폴더를 읽어 다음을 확인했다. 전체 파일 재검증을 완료한 것은 아니다.

| 확인 항목 | Pushing | Inserting |
|---|---|---|
| 폴더 | `data/pushing_sincos_v1/` | `data/inserting_sincos_v1/` |
| 실제 NPZ 수 train / val | 1,800 / 200 | 718 / 82 |
| train 대표 파일의 obs / actions | `(280,12)` / `(280,2)` | `(1588,16)` / `(1588,2)` |
| train·val 각 대표의 `obs[:,:2]+actions-next_obs[:,:2]` 최대 절대 오차 | 약 `5.96e-8 m` | 약 `5.96e-8 m` |
| 기존 KDE·kernel·PF2MP 캐시 후보 | 검출 없음 | 검출 없음 |

구현 시작 시 manifest와 전체 train NPZ를 재확인한다. 관측 필드 순서·단위·정규화 여부, shape·dtype·유한값, 기록 간격, 목표 위치와 행동의 대응, 실제 TCP와의 차이, sin/cos의 타당성, 분할·중복·시연 경계를 확인한다. Val은 정합성 확인에 사용하더라도 KDE support에 포함하지 않는다. 검증 결과는 실험 출력에 저장한다.

Support 후보는 각 train 파일의 `obs[:,0:2]`이며 중첩 window를 열거하여 점을 늘리지 않는다. `next_obs`를 추가해 같은 위치를 중복 계산하지 않는다. 초기 후보는 전체 train 행에서 고정 seed로 최대 3,000점을 균등 추출하는 방식이다. 소수 경로·좁은 통로의 피복을 확인하고 점 수를 확정한다. 3,000점은 논문 Maze의 참고값으로 D3IL에 충분하다고 가정하지 않는다. 긴 시연과 긴 체류 시간에 가중치가 부여되는 추출 규칙도 명시한다.

### 8.5. 추론 시작 시 PDF 캐시 생성과 재사용

사용자 제안을 채택하여 **해당 데이터셋 폴더 안**에 KDE를 재현할 수 있는 표현을 저장한다. PDF는 확률 밀도 함수이며 Gaussian support 점·bandwidth·가중치로 임의 위치의 밀도와 gradient를 다시 계산할 수 있다. Python 객체 pickle 대신 수치 배열과 metadata를 저장한다.

```text
data/pushing_sincos_v1/pf2mp_cache/<fingerprint>.npz
data/inserting_sincos_v1/pf2mp_cache/<fingerprint>.npz
```

NPZ에 support XY, 필요한 가중치, 고밀도 grid 점, grid 설정, JSON metadata를 담고 `allow_pickle=False`로 읽는다. Manifest·원본 NPZ는 수정하지 않으며 `pf2mp_cache/`는 train/val 데이터 탐색 및 Git 추적에서 제외한다.

시작 처리는 `ensure_potential_cache(dataset_dir, density_config)`에 해당하는 공유 함수로 묶어 sample/eval의 policy 구성 전에 한 번만 실행한다.

1. PF2MP off이면 캐시 탐색·생성·train 순회를 하지 않는다. On일 때도 현재 태스크만 대상으로 하며 `data/`의 모든 데이터셋을 자동 생성하지 않는다.
2. Manifest, 정렬한 train 파일의 상대 경로·내용 hash, 특징 정의, Gaussian bandwidth, support 추출 seed·상한, 거리 grid·threshold 설정, 구현 format version으로 fingerprint를 만든다. 초기 구현에서는 내용 hash를 stream으로 계산하며 매번 전체 배열을 로드하는 비용과 구분한다.
3. 해당 NPZ가 있으면 metadata·배열 shape·유한값·저장 내용의 정합성을 검증하고 로드한다. 파일 존재만으로 재사용하지 않는다.
4. 없으면 train을 검증하면서 support와 거리 집합을 구성한다. 고정 seed의 지역 RNG로 재현하고 추론 noise용 RNG를 소비하지 않는다.
5. 동일 fingerprint의 생성 lock을 얻고 존재 여부를 다시 확인한다. 같은 폴더의 임시 파일에 쓴 뒤 atomic rename한다. 동시 실행·중단 시 불완전한 캐시를 완성본으로 읽지 않게 한다. Lock 대기는 timeout과 실패 이유를 제공한다.
6. 손상 캐시는 탐지 사실을 기록하고 격리한 뒤 재생성한다. 데이터·설정 변경은 별도 fingerprint로 저장하고 이전 캐시를 자동 삭제하지 않는다. 읽기 실패·쓰기 권한 없음·용량 부족·lock timeout에서는 이유를 반환하며 빈 밀도장이나 baseline으로 조용히 우회하지 않는다.
7. 검증한 배열은 필요한 device로 한 번 옮겨 episode·재계획·Euler 단계 사이에서 재사용한다. 캐시 경로·fingerprint·artifact hash·hit/miss·생성 시간·로드 시간을 결과에 기록한다.

캐시 식별에는 밀도·거리장을 바꾸는 항목만 넣는다. $\lambda$와 $\alpha$는 query 시 합성하며 checkpoint·normalizer·horizon·Projection/VLSA 플래그와 함께 실험 설정에 기록한다. 이들만 바뀌면 raw XY KDE를 재생성하지 않는다. Threshold·grid·bandwidth·support가 바뀌면 재생성한다. 재사용 시에도 데이터 변경 감지를 위한 hash 순회는 발생하므로 준비 시간과 정상 추론 시간을 분리한다.

### 8.6. on/off, 중복 투영, 모델 호출 수

새 실험의 세 플래그는 `methods.projection.enabled`, `methods.pf2mp.enabled`, `methods.vlsa.enabled`다. CLI에 `--pf2mp` / `--no-pf2mp`를 추가하며 다른 방법을 자동 활성화하지 않는다. PF 수치 설정과 캐시 설정을 구분한다. 의미가 달라지므로 schema v3로 올리고 이전 v2 결과의 읽기를 유지한다.

`methods.pc`는 새 설정의 실행 방법에서 제외한다. 이전 설정·CLI의 PC 지정은 폐기 이유와 함께 거부하고 PF2MP로 암묵 변환하지 않는다. PC 수치·테스트는 이전 커밋으로 재현할 수 있게 기록하고 현재 기본 YAML에서는 이력 주석으로 옮긴다. PC를 주석으로 남겨도 `methods_enabled`나 호출 수에는 포함하지 않는다.

Projection을 `every`로 둔 8가지 조합은 다음과 같다.

| Projection | PF2MP | VLSA | 실행 경로 |
|---|---|---|---|
| off | off | off | 기본 FM |
| on | off | off | FM Euler → 매회 Projection |
| off | on | off | FM+PF Euler |
| off | off | on | FM Euler → terminal VLSA |
| on | on | off | FM+PF Euler → 매회 Projection |
| on | off | on | FM Euler → 매회 Projection → terminal VLSA |
| off | on | on | FM+PF Euler → terminal VLSA |
| on | on | on | FM+PF Euler → 매회 Projection → terminal VLSA |

**중복성 규칙은 PC 때와 달라진다.** PF2MP는 마지막 Euler 이후 독립적으로 실행되지 않으므로 같은 고정 제약·목적·dtype에서 마지막까지 Projection한 직후의 VLSA는 PF on/off에 관계없이 수학적으로 같은 집합에 대한 재투영이다.

- `Projection final + VLSA`는 **PF on이어도 중복 설정으로 거부**한다. PC 때의 중간 보정이 있다는 허용 규칙을 재사용하지 않는다.
- `Projection every + VLSA`는 8가지 비교를 위해 허용한다. 마지막 단계까지 투영했다면 terminal 수정량이 수치 허용 오차 안에서 0인지 확인한다. 별도의 성능 개선 효과를 가정하지 않는다.
- Projection 단독과 VLSA 단독은 생성 중간의 입력이 달라 일반적으로 같지 않다. `final` 비교에 한정하여 동일성을 요구한다.
- PF와 Projection에는 투영의 멱등성을 적용하지 않는다. PF는 분포에 의한 유도이며 Projection은 명시 제약에 대한 최소 수정이다.

모든 조합에서 정상 종료 시 모델 호출은 $N$회, PF 평가는 on일 때 $N$회, PC/CBF-QP는 0회다. 계획 QP 수는 $Q_{projection}+I_{vlsa}$이며 실행 직전 VLSA 재검사는 별도로 집계한다. 초기 비교는 현재 Predict 10회·horizon 16·실행 8회를 유지한다. 이전 PC on의 19회와 새 PF on의 10회를 같은 호출 예산으로 취급하지 않는다.

### 8.7. 최종 검사·실패·계측

현재 `self.enabled=any(method.enabled)`를 그대로 쓰면 PF 단독에서도 기존 QP 제약 위반을 이유로 계획을 거부하여 실험 의미가 바뀐다. 다음 용도를 분리한다.

- `generation_modified`: PF / Projection / VLSA 중 하나가 on. 샘플러 선택과 기록에 사용한다.
- `hard_constraints_enforced`: Projection 또는 VLSA가 on. 기존 초기 workspace 외부 실패와 최종 QP 잔차에 의한 실행 거부에 사용한다.

PF 단독과 all-off에서는 명시 제약을 **측정만** 하며 숨은 투영·잔차에 의한 거부를 넣지 않는다. 비유한값·잘못된 입력 shape·PF 계산 실패는 모든 경로에서 실패로 처리한다. 이로써 8가지 비교에는 강제 처리의 차이도 포함되므로 초기 영역 외부 사례 수·최종 거부율·실행률을 함께 기록하고 필요하면 공통 초기 영역 내부 context에서도 비교한다. 판정 방침 변경을 새 실험 버전으로 남긴다. `evaluation.py`의 `any(method.enabled)` 의존도 같은 의미로 점검한다.

Projection/QP 실패·PF 비유한값은 즉시 종료하며 뒤의 VLSA가 실패 iterate를 복구하지 않는다. VLSA는 유효한 Predict 출력에만 적용한다. 실행 직전 재검사는 기존 VLSA의 명시 옵션으로 유지하고, 행동이 수정되면 실행 후 큐를 폐기한다.

로그는 평탄한 `pipeline_order`에만 의존하지 않고 `pipeline_version`, `predict_update_order: [fm, pf2mp, euler, projection]`, `post_predict_order: [vlsa]`, 활성 플래그를 저장한다. PF는 별도 terminal 단계가 아닌 Predict 내부로 계측한다. 단계별 PF 전후 에너지·gradient norm·PF 증분 norm·Projection 수정량·QP 잔차, 계획 종료 및 VLSA 이후 에너지·잔차, 캐시 준비·PF·모델·QP 시간을 구분한다. Projection 후 에너지가 단조 개선된다고 가정하지 않는다.

### 8.8. 구현 단계와 완료 조건

| 단계 | 주요 대상 | 작업과 완료 조건 |
|---|---|---|
| P0: 기준 고정·데이터 점검 | `data/*/manifest.json`, train NPZ, 기존 policy·결과 | 8.4의 전체 train 확인, checkpoint·normalizer·현재 코드 hash와 고정 noise 기준 저장. 데이터 불일치를 해소한 뒤 밀도장 구성 |
| P1: KDE·거리장·캐시 | 신규 `fm_constraint/pf2mp.py`, 신규 `fm_constraint/tests/test_pf2mp.py` | 전체 train 모집단에서 Gaussian log-density/score, 이산 거리 집합, 위치 scale·채널 마스크, 8.5의 생성·로드·무효화·동시 실행·중단 처리를 구현. CPU에서 구성 가능하도록 함 |
| P2: 설정·PC 중단 | `configs/config.py`, 두 YAML, `test_config.py` | schema v3, PF 설정·CLI·값 검증, PC 지정 거부, final Projection+VLSA 금지 규칙과 명시 제약 강제 조건 구현 |
| P3: Predict 통합 | `policy.py`, `test_policy.py` | 이전 PC 블록을 줄 주석으로 보관. FM+PF→Euler→Projection, 마지막 VLSA. 추가 모델 호출 없이 plan 안에서 KDE를 로드하지 않음 |
| P4: 진입점·기록 | `sample.py`, `evaluation.py`, `eval_*.py`, 집계·렌더링 | 공유 캐시 준비를 한 번만 실행. 새 metadata·시간·실패 판정 기록. 이전 결과는 이전 pipeline으로 읽기 |
| P5: 실모델·환경 확인 | 두 checkpoint, sample, D3IL, 저장 결과 | 8가지 수치 검사, 두 태스크 짧은 rollout, 재계획·큐·이전 행동 확인. 밀도·위치장을 시각화하고 필요 접촉과 금지 접촉 정의 확인 |
| P6: 문서 동기화·본실험 준비 | `FM_Constraint_Integral.md`, `PF2MP.md`, README, 본 PLAN | 이전 PC 단계·19회 호출·조건부 중복 규칙 갱신. 새 검증 summary와 실행 예시 기록. 과거 결과와 분리 |

최소 검증 항목은 다음과 같다.

1. Gaussian 해석 score와 정규화 위치 scale을 포함한 상태 위치 gradient를 유한차분과 대조한다. Action을 바꾸어도 동일 상태의 에너지가 변하지 않고 action guidance가 정확히 0인지 확인한다. 거리항은 최근접점이 유일한 영역과 전환점·거리 0을 구분하여 확인한다.
2. Cache miss→생성→hit에서 밀도장과 고정 noise 계획이 같고 추론 RNG를 바꾸지 않는다. Train·bandwidth 변경으로 무효화되고 $\lambda$만 바꾸면 재사용된다. Val이 support에 들어가지 않으며 손상·동시 생성·쓰기 실패를 처리한다.
3. PF off의 4가지 조합은 현재 PC off의 대응하는 4가지와 계획·상태·호출 수가 같다. All-off는 KDE 파일 유무와 무관하게 baseline과 같다.
4. $\lambda=0$의 유효 설정에서는 밀도장을 준비·기록하되 가산을 엄밀히 0으로 하여 PF off와 같은 noise에서 수치 결과·강제 판정이 같다. 추가 반올림이나 난수를 넣지 않는다.
5. 같은 갱신 전 계획에서 모델과 PF를 평가하고 Euler 후 Projection하는 순서를 확인한다. 상태와 행동이 결합된 모델로 다음 단계에서만 PF의 행동 영향이 생기는지, 한 단계 생성에서는 PF on/off 행동이 같은지 검증한다. Action guidance는 전 단계에서 0이며 현재 관측을 고정하고 추가 모델·CBF-QP 호출이 없음을 확인한다.
6. PF on/off 모두 마지막까지 Projection한 계획에 대한 VLSA는 수치 허용 오차 안에서 멱등성을 만족한다. PF 단독의 제약 위반에는 숨은 거부·투영이 없고 hard on에서만 최종 잔차를 판정한다.
7. 두 실모델의 8가지 조합, `predict_steps=1`, 실제 D3IL에서 재계획 경계를 포함한 짧은 실행, 저장 로그 집계·지도/카메라 재생을 확인한다. 기능 확인과 본평가 성공률을 구분한다.

본실험은 같은 checkpoint·context·noise seed·제약·horizon·실행 chunk로 8가지 조합을 비교한다. PF hyperparameter를 먼저 고정하고 성공률·episode별 금지 접촉률·안전 성공률·제약 위반·계획 거부·다양성·계획 시간을 보고한다. 접촉 관측이 없으면 접촉 지표는 null이며 KDE 밀도나 QP 위반으로 충돌을 추정하지 않는다. PF 계산 비용과 첫 캐시 준비 비용도 보고한다.

구현 진행 상태(아래 기능 검증 완료, 전체 성능 본실험은 별도):

- [x] P0: 전체 train 데이터 확인과 이전 구현 비교 기준 고정
- [x] P1: KDE·거리장·저장/재사용·gradient 검증
- [x] P2: 새 설정과 PC 실행 중단·중복 판정 갱신
- [x] P3: Predict 내부 PF2MP→Euler→Projection 연결
- [x] P4: 모든 추론 진입점·실패 판정·metadata·후처리 대응
- [x] P5: 두 실모델·8가지 조합·실제 환경 기능 확인
- [x] P6: 관련 문서 동기화와 새 실험 기준 확정


### 8.9. 구현 완료 및 검증 근거 (2026-11-06)

검증 요약은 [`pf2mp_validation.json`](../fm_constraint/results/pf2mp_validation.json), 상세 배열·단계 보고·실행 로그·밀도 그림·재생 영상·재현 스크립트는 `fm_constraint/runs/pf2mp_validation/`에 저장했다. 변경 전 기준 커밋은 `e63f75a04a8f22e6b4b191d996eb447a0c63fd21`이다. 기존 사용자 작업과 이전 결과는 보존했다.

- 변경 전 81개 테스트, 새 schema·PF·캐시·실행 경계 테스트 **88개 통과**. Gaussian/거리 gradient 유한차분, action·첫 관측·비대상 채널의 가산항 0, 캐시 재사용·무효화·손상 격리·동시 생성·쓰기 실패·lock timeout·RNG 보존을 확인했다.
- 전체 train 재검증: Pushing **1,800개/417,518전이**, Inserting **718개/711,908전이**. 목표 위치+행동과 다음 목표의 최대 오차는 두 태스크 모두 약 `5.96e-8 m`, sin/cos 제곱합 오차는 약 `1.19e-7`이다. Manifest의 필드·분할·행동 정의·시연 내 연결·시간 간격을 검사했다.
- 두 태스크에 3,000점 support KDE 캐시를 만들고 hit 재사용을 확인했다. 기본 설정의 첫 생성은 이 CPU 환경에서 약 **50초/90초**, hit 준비는 hash 확인을 포함해 약 **0.15~0.2초**였다. 병렬 부하·하드웨어에 따라 달라지는 참고값이다.
- 두 실제 EMA checkpoint에서 **8가지 조합**, 단일 Euler의 **8가지 조합**을 확인했다. PF off의 4가지는 변경 전과 배열 단위로 정확히 같았다. PF on/off 모두 마지막 Projection 뒤 terminal VLSA가 수치적으로 멱등이었다. 모델은 10회, PF guidance는 on일 때 10회, PC/CBF 호출은 0회다.
- 실제 D3IL에서 두 태스크 × 8가지 조합을 각각 **10스텝·2회 계획**으로 실행했다. Sample CLI도 두 태스크에서 각 2개 window를 생성했다. 큐·이전 실제 행동·재계획 경계·실행 결과 저장을 확인했다.
- 집계 및 지도 PNG/GIF, 두 태스크 카메라 PNG/GIF 재생을 확인했다. 명령 재생 오차는 0, 실제 XY 재생 오차는 Pushing 약 `5.85e-7 m`, Inserting 약 `6.20e-7 m`였다. D3IL의 외부 checkout 임시 XML 쓰기는 승인된 실행으로 검증했다.

현재 결과는 **CPU 기능 검증**이다. GPU 실행과 전체 context의 성공률·충돌률은 검증하지 않았다. 접촉 observer가 없는 평가에서 안전 성공률은 계속 null이다. 기본 PF 수치는 초기 후보값이다. 생성한 Inserting 밀도 그림에서는 일부 고정 장벽과 고밀도 영역이 겹치므로, 본실험 전 bandwidth·threshold와 실제 접촉 지표를 별도로 평가해야 한다. 데이터 방문 밀도를 충돌 없는 영역의 증명으로 해석하지 않는다.

현재 실행 예:

```bash
python -m fm_constraint.sample --config configs/pushing.yaml --pf2mp --projection --vlsa \
  --set sample.n_windows=4
MUJOCO_GL=disable python -m fm_constraint.eval_inserting --config configs/inserting.yaml --pf2mp --vlsa
```


## 9. 모델·학습·샘플러 코드 독립화 (2026-11-07)

현재 실험에 필요한 TemporalUnet과 구성 블록, CFM 학습 및 조건 고정, value=None 조건부 Euler를 `fm_baseline/models/`로 이관한다. 모델의 모듈·파라미터 이름, convolution·정규화·attention 계산, CFM의 sigma=0에서도 소비하던 epsilon 난수와 Euler 보폭·detach·chain 순서를 유지한다. 외부 FlowPolicy·QP solver·GuideFlow/SafeFlow/UniConFlow/YFlow는 가져오지 않는다.

학습·baseline sample/eval·통합 sample/eval·16개 순차 실행이 동일한 내부 모델을 import한다. `--hardflow-root`, `HARDFLOW_ROOT` 탐색 및 sys.path 추가를 제거한다. 이전 YAML의 경로 키는 이력으로 읽어 제외하고, 기존 checkpoint의 같은 metadata는 사용하지 않는다. 기존 모델·normalizer·원본 config 및 실험 출력 파일을 수정하지 않는다.

이관 전 실제 두 checkpoint에서 상태 dict 키, 모델 벡터장, Euler sample·chain, CFM loss·대표 파라미터 gradient·학습 RNG, baseline plan·RNG, PF/Projection 네 조합의 계획을 저장하여 이관 후와 비교한다. 외부 패키지 import를 차단한 별도 프로세스에서도 학습·샘플링·실제 두 태스크의 실행을 확인한다. 검증 결과는 `results/local_models_validation.json`에 요약한다.

새 실험 manifest는 내부 모델 및 baseline 로더·trainer의 코드 hash도 포함한다. 기존 본실험의 manifest나 완료 표식은 덮어쓰지 않는다. 이관 전 동결 manifest의 재개가 코드 변경을 거부하는 것은 의도된 동작이며, 이전 실행의 재개는 원래 코드 snapshot, 이관 코드의 새 실행은 새 출력 경로를 사용한다. 이전 결과의 집계·재생은 계속 가능하다.

### 9.1. 완료 기록

- [x] 필요한 모델 구성만 내부 모듈로 이관하고 출처·원본 파일 hash·기준 커밋을 기록했다. 이관 대상 원본 클래스는 로컬 수정본에서도 기준 커밋의 클래스와 같음을 AST로 대조했다.
- [x] 두 실제 EMA checkpoint의 키·벡터장·Euler sample/chain·CFM loss·대표 gradient·RNG·baseline 계획·PF/Projection 네 조합의 계획이 이관 전후 **배열 단위로 정확히 일치**했다.
- [x] **97개 테스트 통과**. 외부 패키지를 차단한 새 프로세스에서 두 checkpoint를 읽고 상태·행동 shape과 CFM 수식·gradient·RNG를 확인했다.
- [x] 외부 HardFlow 형제 폴더가 없는 `/tmp` 격리 복사본에서 두 태스크의 PF+Projection 샘플링을 확인했다. 데이터·checkpoint만 입력으로 연결하고 외부 패키지 namespace를 차단했다.
- [x] 내부 모델로 실제 학습 2 update와 baseline/통합 sample CLI를 확인했다. 원본 checkpoint·normalizer의 hash는 그대로다.
- [x] 외부 import를 차단한 실제 D3IL에서 두 태스크 × 8개 조합을 각각 1개 context·2개 seed·최대 2 step으로 실행하고 집계까지 완료했다. 기존 본실험 결과는 수정하지 않았다.

검증 요약은 [`local_models_validation.json`](../fm_constraint/results/local_models_validation.json), 상세 배열·이관 전 원본·독립 실행 스크립트·16개 짧은 실행 결과는 `fm_constraint/runs/hardflow_independence/`에 저장했다. CPU 기능 검증이며 GPU와 전체 태스크 성공률은 다시 평가하지 않았다. 시뮬레이터용 D3IL checkout은 계속 필요하다.
