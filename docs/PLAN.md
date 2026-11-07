# 브랜치 통합 계획 (2026-11-07)

목표: 모든 브랜치를 `master` 하나로 합친다. 우선순위는 다음 순서다.

1. `feature/method123_merge`의 **HardFlow 의존성 제거**를 master에 반영한다.
2. train-free 5개 방법 **① Projection, ② PF2MP, ③ Executed-action filter, ④ RTC, ⑤ LiPo**를 하나의 실행 경로에서 독립적으로 켜고 끌 수 있게 한다.
3. legacy 코드와 불필요한 테스트를 삭제한다.

> 이전 PLAN(Projection → PC → VLSA, PF2MP 교체, 모델 독립화 기록)은 `git show b577ae9:docs/PLAN.md`로 볼 수 있다. 이 문서에서는 다시 옮겨 적지 않는다.

---

## 0. 현재 상태 (fetch 직후)

공통 merge-base는 `2442641`(method2 결과 push)이다.

| 브랜치 | master 대비 (뒤처짐/앞섬) | 내용 | HardFlow |
|---|---|---|---|
| `origin/master` (`cba3f3f`) | 기준 | `fm_baseline` + LiPo·smoothness, `fm_pc`(PC + ACG + OOD + 6개 추가 태스크), `ood/`, `visualizer/`, `experiments/`, `tests/` | **사용** |
| `feature/method123_merge` (`b577ae9`) | 13 / 16 | `fm_baseline/models/`로 모델 내재화, `fm_constraint/`(①②③), `configs/`, `scripts/run_method123.sh`. `fm_pc/`는 삭제 | **제거 완료** |
| `codex/rtc-qp-method123` (`d001c1e`) | 0 / 4 | `fm_baseline/rtc.py`(④), `lipo.py`에 `mode="qp"` 추가(⑤), `run_qp123*.py`. 예전 `fm_constraint`를 **파일 복사**로 가져오고 `fm_pc/`·`ood/`를 삭제함 | 사용 |
| `feature/rtc` (`25f536a`) | 0 / 3 | 위 codex 브랜치의 앞 3커밋과 같음 (codex 브랜치에 포함됨) | 사용 |
| `feature/method4` (`183ebc2`) | – | `fm_l2c2/`(L2C2 **학습 기반** 정규화). RTC를 추가했다가 revert함 | 사용 |
| `feature/method1/2/3` | – | `method123_merge`에 이미 merge됨 | – |
| `visualizer` (`c7c64bc`) | – | master에 이미 포함된 것으로 보임. 4단계에서 `git branch --contains`로 확인 | – |
| `feature/ood-benchmark` | – | master에 PR #3으로 merge됨 | – |

`git merge-tree`로 미리 본 `method123_merge ← master` 충돌은 다음과 같다.
- 내용 충돌: `.gitignore`, `README.md`, `fm_baseline/eval_inserting.py`, `fm_baseline/eval_pushing.py`
- modify/delete 충돌: `fm_pc/{README.md, constraints/__init__.py, build.py, inject.py, eval_inserting.py, eval_pushing.py}`. HEAD는 삭제했고 master는 수정했다.

HEAD 테스트는 `pytest fm_constraint/tests`에서 **102개 모두 통과**한다.

### 발견한 문제
- **HEAD 버그**: `fm_baseline/eval_inserting.py:140`이 정의되지 않은 `cfg0`를 사용한다. master에는 정의가 있으므로 merge할 때 함께 고친다.
- **③의 정의가 브랜치마다 다르다**:
  - `method123_merge`와 `docs/report.md`: ③은 실행 직전 H=1 QP(`execution_filter`)이고, terminal VLSA는 OFF.
  - codex RTC 브랜치: ③은 terminal-plan VLSA이고, 실행 직전 재검사는 OFF.
  - report.md의 정의(실행 직전 필터)를 기준으로 통일한다.
- **⑤ LiPo도 정의가 갈린다**: master의 LiPo(blend + minimum-jerk, clarabel)와 codex의 `mode="qp"` 고정 jerk QP(0.3 mm)가 있다. → codex의 고정 jerk QP로 확정했다(9절).
- `pushing.yaml`의 `paths.contexts_file`이 `inserting_val.pkl`을 가리킨다. Pushing에서 실제로 쓰이는지 확인 후 정리한다.
- `fm_pc/benchmark.py:6`이 master에 없는 `check_benchmark.py`를 참조한다.

---

## 1. 통합 방식

- **작업 브랜치**: `method123_merge`에서 `integrate/all-methods`를 만든다. 원격 `master`에는 직접 commit하지 않는다. 최종적으로 PR 하나(또는 단계별 PR)로 master에 반영한다.
- **기준 코드**: 패키지 구조는 `method123_merge`(`fm_baseline/models` + `fm_constraint` + `configs`)를 기준으로 한다. master의 기능은 이 구조로 옮겨 온다.
- **git 이력**:
  - master는 진짜 `git merge`로 합친다. 이력을 보존하고 PR이 가능하다.
  - codex/rtc 브랜치는 `fm_constraint`를 별도로 복사해서 갈라졌기 때문에 그대로 merge하면 같은 파일이 이중으로 충돌한다. 그래서 **RTC 관련 파일만 가져와 다시 연결**한다(`git checkout <ref> -- fm_baseline/rtc.py` 등). 이후 `git merge -s ours`로 "이미 반영됨"이라는 merge 기록만 남긴다.
  - method4(L2C2)는 삭제하기로 했으므로 merge하지 않는다(9절).
- 단계마다 commit하고, 단계 끝에서 테스트와 smoke를 실행한다.

---

## 2. 1단계: master merge + HardFlow 제거 (최우선)

1. `git switch -c integrate/all-methods feature/method123_merge && git merge origin/master`
2. 충돌 해결 규칙:

| 파일 | 해결 방법 |
|---|---|
| `fm_baseline/train.py`, `sample.py` | HEAD의 `from fm_baseline.models import ...`를 유지한다. master의 `TASK_DIMS` 확장(avoiding/aligning/sorting/stacking)은 받는다. `resolve_hardflow`와 `--hardflow-root`는 삭제한다 |
| `fm_baseline/eval_pushing.py`, `eval_inserting.py` | master의 기능(LiPo postprocess, `EvaluationDiagnostics`, `--context-start`, `--record-joints`, `--capture-*`, `--replan-noise`, `generator=`)을 기준으로 하고, `load_model`은 HEAD의 4인자 시그니처로 맞춘다. `cfg0` 버그를 고친다 |
| `fm_baseline/eval_smoothness.py`, `compare_smoothness.py` | provenance의 `hardflow_commit`을 `model_implementation`과 models 파일 hash로 바꾼다. 예전 결과와의 호환은 두지 않는다 |
| `fm_baseline/run_lipo_comparison.py`, `run_lipo_sharded.py` | `--hardflow-root` 전달을 삭제한다 |
| `fm_pc/*` (modify/delete) | **일단 master 버전을 복원**한 뒤 2단계에서 `fm_constraint`로 옮기고 삭제한다. 복원한 직후에 HardFlow import(`fm_pc/eval_pushing.py:57-63`의 `load_model` 복사본 등)를 `fm_baseline.models`/`fm_baseline.eval_pushing.load_model`로 바꾼다 |
| `visualizer/app.py`, `static/index.html`, `static/app.js` | `hardflow_root` 필드, 입력란, CLI 전달을 제거한다 |
| `tests/test_lipo_evaluation.py`, `tests/test_visualizer.py` | `hardflow_root` 인자를 제거한다 |
| `.gitignore`, `README.md` | 두 쪽 내용을 합친다. README의 설치 안내에서 HardFlow를 빼고 `THIRD_PARTY_NOTICES.md`를 참조하게 한다 |
| `results/*_baseline/config.json`의 `hardflow_root` | 키를 삭제한다. checkpoint 가중치는 그대로 둔다. `configs/config.py`의 `hardflow_root` pop 호환 코드도 함께 삭제한다 |

3. **완료 조건**
   - `git grep -in hardflow -- ':!docs' ':!THIRD_PARTY_NOTICES.md' ':!*.json'`의 결과가 0건이다(호환 처리 코드만 예외).
   - `pytest fm_constraint/tests tests/`가 통과한다.
   - `HARDFLOW_ROOT`가 없는 환경에서 `fm_baseline.sample`과 `fm_constraint.sample`이 동작한다.
   - all-off 계획이 `fm_constraint/results/local_models_validation.json`의 기준과 배열 단위로 일치한다.

---

### 1단계 결과 (2026-11-07, 브랜치 `integrate/phase1-hardflow`)
- 해결한 충돌: `.gitignore`, `README.md`, `fm_baseline/eval_{pushing,inserting}.py`. `fm_pc/`는 master 버전 전체를 복원했다(HEAD가 지운 `constraints/{pc,qp,sets}.py`, `check_constraints.py`, `compute_metrics.py`, `render_rollouts.py` 포함). `fm_pc/results/`는 `fm_constraint/results/legacy/pc`와 중복이라 복원하지 않았다.
- HardFlow 제거:
  - `fm_baseline`: `eval_smoothness`(provenance를 `model_source_sha256`로 바꿈), `compare_smoothness`, `run_lipo_*`, `train`
  - `fm_pc`: `eval_*`, `check_*`, docstring
  - `visualizer`: app, UI, README
  - `tests`
  - `configs/config.py`의 호환 pop, `results/*_baseline/config.json`의 키
  - `fm_constraint/tests/test_local_models.py`의 호환 테스트
- `eval_inserting.py`의 `cfg0` 버그를 고쳤다.
- 검증:
  - `pytest fm_constraint/tests tests`: **119 passed**. visualizer 테스트 2개는 실행 환경에 `fastapi`가 없어 실행하지 못했다.
  - master + HardFlow와 merge 결과의 baseline 계획(`Policy.plan`, 두 태스크 × 4개, 고정 generator)이 **비트 단위로 일치**한다.
  - D3IL smoke(24 step)가 통과했다: `fm_baseline` pushing + LiPo, inserting, `fm_pc/eval_task` OOD `A_mid`, `fm_constraint.eval_inserting` PF2MP + Projection.
  - 남은 HardFlow 언급은 출처 표기(README, `fm_baseline/README.md`, `THIRD_PARTY_NOTICES.md`, `requirements.txt` 주석)와, HardFlow import를 차단하는지 검사하는 테스트뿐이다.

## 3. 2단계: master의 fm_pc 기능 이관

`fm_pc`의 PC Correct는 schema v3에서 이미 폐기됐다. 남길 가치가 있는 것은 OOD 평가 기반과 추가 태스크다.

| master 모듈 | 처리 |
|---|---|
| `fm_pc/tasks.py` (TaskSpec 8종) | `fm_constraint/tasks.py`로 옮기고 8개 태스크를 모두 유지한다. 5개 방법의 제약 설정(workspace·가속도·envelope)은 Pushing·Inserting만 있으므로, 추가 태스크는 baseline과 OOD 평가가 동작하는 것까지를 이번 범위로 한다 |
| `fm_pc/ood.py`, `ood_metrics.py`, `make_ood_dataset.py` | 3.1절에 따라 `ood/` 패키지로 합친다. codex 브랜치의 `fixed_conditions.py`도 여기에 흡수한다 |
| `fm_baseline/ood_benchmark/*` (pkl + manifest) | 3.1절에 따라 JSONL로 변환해 `data/ood/`로 옮긴다 |
| `fm_pc/constraints/acg.py`, `check_acg.py` (ACG) | **삭제** |
| `fm_pc/benchmark.py`, `eval_task.py` | `fm_constraint/evaluation.py`의 태스크 일반화에 흡수한 뒤 삭제한다 |
| `fm_pc/constraints/{pc,sets,qp,build,inject}.py`, `check_constraints.py`, `compute_metrics.py`, `render_rollouts.py` | `fm_constraint`에 이미 대응하는 코드가 있으므로 **삭제**한다 |
| `preprocess/prepare_task.py`, `check_prepare_task.py` | 유지 |

완료 조건: `fm_pc/` 디렉터리를 삭제한 뒤에도 Pushing·Inserting OOD 조건 1개의 smoke 평가가 통과한다.

### 3.1 OOD 두 구현 합치기

두 사람이 같은 A/B/C 요인 설계를 서로 다른 태스크·형식으로 구현했다.

| | 구현 1 (`feature/ood-benchmark`, PR #3) | 구현 2 (`cce3c39`) |
|---|---|---|
| 코드 | `fm_pc/ood.py`, `ood_metrics.py`, `make_ood_dataset.py`, `tasks.py`, `eval_task.py` | `ood/generate.py`, `verify.py`, `validate_sim.py`, `summarize_sim.py` |
| 데이터 | `fm_baseline/ood_benchmark/{pushing,inserting,aligning}/*.pkl` + `manifest.json` | `data/ood_sas_v1_20261106/{sorting2,4,6,stacking,avoiding}.jsonl` |
| 태스크 | Pushing(60), Inserting(82), Aligning(60) | Sorting-2/4/6, Stacking, Avoiding (조건당 dev 20 / test 100) |
| 조건 | A 블록 초기 위치, B 로봇 시작, C 관측 노이즈. weak/mid/strong + AB/AC/BC/ABC_mid + id (14개) | A 물체 위치, B 로봇 시작, C 노이즈. 태스크별로 7~16개 조건 |
| 그 외 | 실행 중 push·inference delay 섭동, seam/in-chunk jerk 지표, 평가기 연결 완료 | sha256·분리 조건 검증, MuJoCo 초기화 스크리닝. `rollout_ready=false`, 즉 평가기와는 미연결 |

태스크는 겹치지 않고 서로 보완적이다. 합치는 방법:
1. **하나의 패키지**로 모은다: `ood/`. 생성은 `ood/generate.py`, 검증은 `ood/verify.py`와 `validate_sim.py`, 평가 시 적용은 `ood/apply.py`(현재 `fm_pc/ood.py`의 shift/noise/push/delay), 지표는 `ood/metrics.py`가 맡는다. `fm_pc/make_ood_dataset.py`는 `generate.py`에 흡수한다.
2. **하나의 데이터 형식**: 사람이 diff로 검토할 수 있는 JSONL(구현 2 형식)로 통일한다. 8개 태스크 전부를 아래 결정에 따라 새로 생성한다. 위치는 `data/ood/<task>.jsonl` + `manifest.json`이다. 기존 `ood_benchmark/*.pkl`과 `data/ood_sas_v1_20261106/`은 새 데이터의 검증이 끝나면 삭제한다(git 이력에 보존).
3. **하나의 조건 이름 체계**: `id`, `{A,B,C}_{weak,mid,strong}`, `{AB,AC,BC,ABC}_mid`. 구현 2에서 태스크별로 다른 조건(예: avoiding에는 A 없음)은 manifest에 태스크별 허용 조건 목록으로 둔다.
4. **하나의 평가 진입점**: `fm_constraint` 평가기에 `--ood <condition>` 하나로 연결한다. 관측 노이즈 seed와 정책 seed는 방법 간에 paired로 맞춘다(두 구현의 seed 규칙 중 하나로 통일하고 manifest에 기록).
5. **검증**: 두 구현 각자의 테스트(`tests/test_ood_generation.py` 등)를 통합 형식에 맞춰 유지한다. 태스크마다 조건 1개씩 MuJoCo reset smoke를 통과해야 한다.
6. 통합 형식(특히 seed 규칙과 조건 강도 정의)은 **두 구현자의 확인을 받은 뒤** 확정한다.

#### 결정 (2026-11-07)

- **조건 강도는 구현 1을 따른다.**
  - A: 모든 블록을 시연 초기 범위 밖 d로 이동한다.
  - B: 임의 방향으로 이동하고, 시뮬레이터에서 경로와 도착 위치를 검증한다.
  - C: xy와 yaw 노이즈를 함께 적용한다.
  - 조건 목록은 `id`, `{A,B,C}_{weak,mid,strong}`, `{AB,AC,BC,ABC}_mid`의 14개로 한다.
  - 구현 2의 태스크(sorting 2/4/6, stacking, avoiding)도 이 정의로 다시 생성한다.
  - 거리와 σ 수치는 태스크 기하에 맞춘다. 기본값은 구현 2의 수치(sorting A 3/6/10 cm, stacking 1.5/3/6 cm, C xy 3/6/12 mm)이고, 모든 블록을 옮겼을 때 배치가 불가능하면 aligning처럼 가능한 최대값으로 낮춘다. 낮춘 경우는 manifest에 기록한다.
  - avoiding처럼 요인이 없는 태스크(A 없음)는 manifest의 태스크별 허용 조건 목록에서 해당 조건을 제외한다.
  - stacking의 손끝 yaw(±5/10/20°)는 OOD에서 **제외**한다(2026-11-07 결정). B는 XY 오프셋만 적용한다.
- **seed 규칙은 구현 2를 따른다.**
  - 데이터 생성: `seed_for(root=20261106, task, split, i, …)` 해시를 쓰고, dev/test split의 seed를 분리한다. 구현 1의 common random numbers(조합 조건이 단일 요인의 배치를 재사용)는 해시 키를 요인 단위로 잡아 같은 성질을 유지한다. 예를 들어 `seed_for(root, task, split, i, 'A')`는 A_mid와 AB_mid에서 같은 값을 쓴다.
  - 관측 노이즈: `(case seed, rollout, replan, entity, component)` 해시로 값마다 독립적으로 생성한다. 재계획 횟수가 달라도 같은 재계획 시점의 노이즈가 같다.
  - 정책 noise: 사례마다 파일에 저장한 `policy_seeds`(기본 4개)를 사용한다. 따라서 `fm_constraint` 평가기의 `sha256([task, context_id, sampling_seed])` 규칙과 `--seeds` 목록은 OOD 평가에서 `policy_seeds[rollout]`로 대체한다. ID 평가도 같은 형식(`id` 조건 파일)으로 통일한다.
- pushing, inserting, aligning의 기존 pkl은 변환하지 않고, 위 규칙으로 **다시 생성**한다(seed 규칙이 바뀌므로 기존 배치와 같아지지 않는다). 기존 pkl로 얻은 결과는 v1로 표시하고 새 결과와 섞지 않는다.

#### 팀원 확인 사항 (참고: 비교 당시 자료)

**Q1. 조건 강도 정의가 다르다. 어느 쪽으로 맞출까?**

| | 구현 1 (pushing·inserting·aligning) | 구현 2 (sorting·stacking·avoiding) |
|---|---|---|
| A 물체 위치 | **모든 블록**을 시연 초기 범위 밖 d로 이동. d = 2.5/5/10 cm (aligning 5/10/12) | **한 블록만** 초기 영역 밖 d로 이동. d = 3/6/10 cm (stacking 1.5/3/6) |
| B 로봇 시작 | 2/5/10 cm, 임의 방향, 시뮬레이터에서 경로 검증 | 2/5/10 cm, 8방향, stacking은 yaw ±5/10/20° 추가 |
| C 관측 노이즈 | xy σ 2.5/5/10 mm와 yaw 2.5/5/10°를 **함께** 적용 | xy σ 3/6/12 mm와 yaw 2.5/5/10°를 **각각 다른 조건**으로 분리 |
| 조합 조건 | AB/AC/BC/ABC_mid 포함 (14개) | 단일 요인만 (sorting 13개) |

- 태스크마다 기하가 달라서 수치 자체는 달라도 된다. 다만 **A가 "모든 블록"인지 "한 블록"인지**, **C의 xy/yaw를 함께 줄지 따로 줄지**, **조합 조건을 넣을지**는 통일해야 태스크 간 결과를 같은 축으로 비교할 수 있다.

**Q2. seed 규칙이 다르다. 어느 쪽으로 맞출까?**

| | 구현 1 | 구현 2 |
|---|---|---|
| 데이터 생성 | `default_rng([seed=0, context, 요인])`. 요인별 독립 stream이라 조합 조건이 단일 요인의 배치를 재사용한다(common random numbers) | `seed_for(20261106, task, split, i, …)` 해시 기반. dev/test split의 seed를 분리 |
| 관측 노이즈 | episode별 `noise_seed` + trajectory로 만든 **하나의 stream**에서 재계획마다 순서대로 뽑음 | `(case seed, rollout, replan, entity, 좌표)` 해시로 **값마다 독립적으로** 생성. 호출 순서와 무관하게 재현됨 |
| 정책 noise | 평가기의 `--ood-seed`, context, trajectory로 결정 (파일에 저장하지 않음) | 사례마다 `policy_seeds` 4개를 **파일에 저장** |

- 참고: 현재 `fm_constraint` 평가기의 정책 noise는 `sha256([task, context_id, sampling_seed])`이다.
- 제안: 관측 노이즈는 구현 2 방식(값별 해시)을 쓴다. 방법마다 재계획 횟수가 달라도 같은 재계획 시점의 노이즈가 같게 유지되기 때문이다(RTC, ③ 필터의 큐 폐기가 재계획 횟수를 바꾼다). 정책 seed는 평가기 규칙 하나로 통일한다(파일에 저장된 `policy_seeds`는 사용하지 않음).

---

### 2단계 결과 (2026-11-07, 브랜치 `integrate/phase2-ood`)
- **삭제**:
  - `fm_pc/` 전체(PC, ACG, OOD v1 코드 포함)
  - `ood/`의 구현 2 도구(`validate_sim.py`, `summarize_sim.py`, 이전 `generate.py`/`verify.py`)
  - `data/ood_sas_v1_20261106/`, `fm_baseline/ood_benchmark/`
  - `tests/test_ood_generation.py`
- **이관**:
  - `fm_pc/tasks.py` → `fm_constraint/tasks.py`. 8개 태스크를 유지하고, 이전 OOD API를 쓰던 `Avoiding.shift_start`는 삭제했다.
  - 추가 태스크용 baseline 평가기 → `fm_baseline/eval_task.py`(ID만 지원)
  - `preprocess/check_prepare_task.py`의 import를 새 위치로 바꿨다.
- **새 `ood/` 패키지**: `conditions.py`, `apply.py`, `generate.py`, `verify.py`, `metrics.py`, `README.md`.
  - 조건은 20개다(A/B/C 14개 + D push 3개 + E 지연 3개).
  - D·E 강도 기본값은 **D 2/4/8 cm(chunk 중간), E 1/2/4 step**이다. 확정 전이므로 `ood/conditions.py`의 `LEVELS`에서 조정한다.
- **평가 연결**: `fm_constraint.eval_* --ood <조건>`(`eval.ood`, `paths.ood_root`).
  - 정책 seed는 파일의 `policy_seeds`를 쓴다.
  - C 노이즈는 정책과 실행 직전 필터에만 넣고, 위반 측정은 실제 상태로 한다.
  - D push는 남은 큐가 4개일 때 적용한다.
  - E 지연에서는 계획을 미리 세우고 index d부터 실행한다. ③이 행동을 수정하면 대기 중인 계획도 폐기한다.
  - 기록: rollout마다 `seam`과 `ood_events`를 남긴다.
- **데이터** `data/ood/`(v2): Pushing 60 context × 20 = 1,200행, Inserting 82 × 20 = 1,640행.
  - A의 배치는 구현 1의 기존 pkl과 모든 context에서 일치한다(결정적 규칙).
  - A·B의 실패(옮기지 못한 블록, 시작 위치를 찾지 못한 경우)는 0건이다.
  - `python -m ood.verify` OK.
- **검증**:
  - `pytest fm_constraint/tests tests`: 127 passed(visualizer 2개 제외, fastapi 없음).
  - D3IL smoke 통과:
    - Pushing `id`, `B_mid`, `C_mid`+①, `D_mid`(step 84 push), `E_mid`(계획 30회 → 32회)
    - Inserting `ABC_mid`+①②, `D_strong`(step 300 push)
    - `ood.metrics` 집계, `fm_baseline/eval_task.py`(pushing)
- **제외**: 구현 1의 학습 관측 최근접 거리(`OOD dist`) 지표. 계획 관측을 저장하지 않기 때문이다.

## 4. 3단계: 5개 방법 하나의 파이프라인으로 연결

### 4.1 설정 (schema v4)
`configs/config.py`의 `methods`를 다음 5개 키로 고정한다. 각 키는 `enabled`를 가진다.

```yaml
methods:
  projection: {enabled, mode: every, skip_first}            # ①
  pf2mp:      {enabled, weight, bandwidth_m, ...}            # ②
  filter:     {enabled, invalidate_queue_on_change}          # ③ 실행 직전 H=1 QP
  rtc:        {enabled, beta: 5.0, delay_steps: 0}           # ④
  lipo:       {enabled, bound_m: 0.0003}                     # ⑤ 고정 jerk QP (codex mode="qp")
```
- 지금의 `methods.vlsa`(terminal-plan)와 `vlsa.execution_recheck`, `Projection final + VLSA` 거부 분기를 삭제한다. 실행 직전 필터는 `methods.filter` 하나로 표현한다(10절).
- `--execution-filter` CLI와 method123 runner의 `executed_action_filter` 플래그를 `methods.filter`로 통일한다.
- 조합 ID는 5비트로 정한다: ①=16, ②=8, ③=4, ④=2, ⑤=1 (`00`~`31`). 이렇게 하면 기존 `P00 P04 … P28`(④⑤ OFF)과 번호가 그대로 맞는다.

### 4.2 실행 순서 (한 번의 재계획)
```text
[④ RTC] 직전 계획의 미실행 suffix → soft-mask target
Predict N회: v = FM(x,t) [+ ④ RTC guidance] [+ ② PF2MP] → Euler → [① Projection]
→ 계획 확정 → [⑤ LiPo: jerk QP, 이전 실행 history 고정]
→ 실행 큐에서 하나씩: [③ filter: 최신 관측 H=1 QP] → env.step
→ 앞 K=8개 실행 후 재계획
```
결정할 규칙(구현할 때 테스트로 고정한다):
- ④의 참조 suffix는 **⑤ 처리 후 실제로 큐에 있던 행동**을 쓴다. codex 브랜치와 같다.
- ③이 행동을 수정하면 남은 큐를 폐기하고 즉시 재계획한다. 이때 ④의 참조 suffix는 비워서 다음 계획을 RTC 없이 만든다. 위 두 규칙은 결과 metadata에 기록한다.
- ①/③이 켜져 있으면 ⑤ 이후의 행동이 제약을 다시 위반할 수 있다. 그래서 ⑤는 **제약 투영 전**에 적용한다. 즉 순서는 ⑤ → 최종 검사 → ③이고, 이때 ⑤의 jerk QP에 ① 제약을 함께 넣을지는 2차 과제로 둔다.
- 첫 계획에는 ④·⑤ 모두 참조가 없으므로 no-op이다. episode reset 때 내부 상태를 초기화한다.

### 4.3 코드 위치
- `fm_baseline/rtc.py`: codex에서 가져온다. HardFlow 의존이 없는 순수 numpy/torch 코드다.
- `fm_baseline/lipo.py`: codex의 `mode="qp"` 경로(`minimum_jerk` + `frozen_prefix`)만 남긴다. blend 경로와 `ChunkPostprocessor`의 blend 분기, `--postprocess blend`는 삭제한다.
- `fm_constraint/policy.py`: `plan(..., rtc_suffix=None)`를 받고 Predict 루프 안에 RTC guidance를 넣는다. codex 브랜치의 `policy.py:90-170`을 참조한다.
- `fm_constraint/evaluation.py`: ⑤ 후처리, ③ 필터, ④ suffix 관리를 한 rollout 루프에서 처리한다.
- `fm_constraint/method123.py` → `fm_constraint/suite.py`(32조합 runner)로 바꾼다. `scripts/run_method123.sh` → `scripts/run_suite.sh`로 바꾼다.

### 4.4 검사 ("자연스럽게 작동"의 기준)
1. **단위 테스트**
   - 각 방법을 OFF로 하면 baseline과 비트 단위로 같다.
   - 각 방법을 단독으로 켜면 해당 단계만 실행된다(호출 수와 report 키로 확인).
   - RTC soft mask의 경계 조건, LiPo history 고정, filter의 큐 폐기를 확인한다.
2. **32조합 설정 해석 테스트**: 모든 조합에서 config 해석, `pipeline_order` metadata, 유한값이 정상이다.
3. **오프라인 smoke**: 두 태스크의 실제 checkpoint로 32조합을 각각 2회 재계획(sample 진입점)한다.
4. **D3IL smoke**: 두 태스크 × 32조합 × 1 context × 1 seed, `--max-steps` 단축. 실패, 계측 누락, 재계획 경계를 확인한다.
5. **비교 기준 일치**: ④⑤ OFF 8조합이 기존 method123 runner의 결과와 같은 seed에서 일치한다(코드 변경 전후).

---

### 3단계 결과 (2026-11-08, 브랜치 `integrate/phase3-methods`)
- **schema v4**: `methods.{projection, pf2mp, filter, rtc, lipo}`. CLI `--projection --pf2mp --filter --rtc --lipo`.
  - terminal VLSA와 `execution_recheck`, PC 주석 블록, PC·v2 호환 코드를 삭제했다.
  - 조합 비트는 ①16 ②8 ③4 ④2 ⑤1이다.
- **④ RTC** (`fm_constraint/rtc.py`, codex에서 가져옴): Predict의 각 스텝 벡터장에 `guided_velocity`를 쓴다(beta 5).
  - 참조는 활성 계획의 미실행 H−s개 행동이다.
  - 지연 d가 있으면 soft mask의 앞 d개를 1로 둔다.
  - ③이 행동을 수정한 직후 계획은 참조 없이 만든다.
- **⑤ LiPo** (`fm_constraint/lipo.py`): codex `mode="qp"`와 같은 목적함수다. 결정 1-a에 따라 ①이 켜져 있으면 ① 제약을 같은 QP에 넣는다. 지연 중 실행이 확정된 d개 행동은 등식으로 고정한다.
  - **solver 변경**: 4.3절과 11절의 "OSQP 하나로 통일"과 달리 ⑤만 **Clarabel(interior point)**을 쓴다.
    - OSQP로 풀면 반복 한도에 걸리거나 한도를 1e-9 m 넘겼고, 계획당 약 370 ms가 걸렸다.
    - 원래 master·codex LiPo도 Clarabel을 썼다. ①·③은 그대로 OSQP다.
  - **닿을 수 없는 ① 면 제외**: 보정 한도 안에서는 닿을 수 없는 면(여유 > ‖G_row‖₁·2·bound)을 빼고 푼다(해는 같다). 704면을 모두 넣으면 Clarabel이 `InsufficientProgress`로 멈췄다. 최종 잔차는 전체 면으로 판정한다.
- **runner**: `method123.py` → `suite.py`(32조합, `P00`~`P31`·`I00`~`I31`), `method123_metrics.py` → `suite_metrics.py`, `run_method123.sh` → `run_suite.sh`.
  - Complementarity 쌍은 실행 전에 고정했다: 가장 낮은 비트 방법이 B, 나머지가 A(기존 12/20/24/28 쌍과 같다).
  - `run_inserting_gpu_group.sh`와 그 테스트는 삭제했다(5절 목록).
  - runner에 OOD 조건을 넣는 일은 실험 환경 통일 단계로 미룬다.
- **⑤의 범위**: `fm_baseline/lipo.py`의 blend 경로 삭제는 4단계(정리)에서 한다. 현재 ⑤는 `fm_constraint/lipo.py`이고, `fm_baseline/lipo.py`는 baseline 평가 스크립트의 `--postprocess`에서만 쓴다.
- **검증 (4.4)**:
  1. pytest 135 passed(visualizer 2개 제외). 범위:
     - ①②④⑤ 16조합의 호출 수와 조건 유지, ④ 참조 유무와 beta=0 동등성
     - ⑤의 고정 구간 유지, ① 잔차, 실패 시 중단
     - ③ 큐 폐기 후 ④ 참조 제거, 지연 중 고정 구간·suffix·history 전달
     - 32조합 비트와 비교 쌍
  2. 2태스크 × 32조합 D3IL smoke(24 step)를 모두 통과했다. Clarabel로 바꾸기 전에는 Pushing의 ①+⑤ 8조합이 첫 계획에서 실패했다.
  3. ①+⑤ 스트레스: Pushing 17·31, Inserting 31 조합 각 10 context × 600 step에서 LiPo 실패 0건, 계획당 최대 약 50 ms.
  4. 지연 E_strong + 5개 모두 ON, 두 태스크 × 2 context × 200 step:
     - 계획 52회 중 ④ 참조 50회, 고정 구간 50회, ⑤ 성공 52회
     - 위반 step 0
  5. **④⑤ OFF 8조합 × 2태스크의 실행 행동이 2단계 코드와 비트 단위로 같다**(D3IL, 24 step, seed 0).

## 5. 4단계: 삭제 목록 (legacy / 불필요 테스트)

삭제는 각 단계의 대체 기능이 검증된 뒤 **별도 commit**으로 한다. 결과 JSON처럼 재현의 근거가 되는 파일은 git 이력으로 보존된다는 점을 확인한 뒤 지운다.

| 대상 | 이유 |
|---|---|
| `fm_pc/` 전체 | PC 폐기. 쓸모 있는 부분은 2단계에서 `fm_constraint`로 이관 |
| `fm_constraint/policy.py`의 `LEGACY_PC` 주석 블록(약 60줄) | git 이력에 남아 있음 |
| `configs/config.py`의 PC 거부 분기와 `hardflow_root` pop, schema v2 결과 읽기(`compute_metrics`, `render_rollouts`) | 이전 설정 호환을 두지 않으므로 삭제 |
| `fm_constraint/results/legacy/` | 2026-10-22 PC/Projection 이관 결과. 이력으로 충분 |
| `fm_constraint/results/order_projection_pc_vlsa_validation.json`, `integration_*.json`, `method123_runner_validation.json` | 폐기된 순서와 runner의 검증 기록 |
| `fm_constraint/tests/test_gpu_group_script.py`, `scripts/run_inserting_gpu_group.sh` | method123 전용 GPU 분할. suite runner로 대체 |
| `fm_constraint/tests/test_local_models.py` | HardFlow와의 비교 테스트. 비교 대상이 사라짐(결과 JSON만 보존) |
| ACG 전체(`acg.py`, `check_acg.py`, `build.py`/`inject.py`의 ACG 인자) | 결정 1 |
| `fm_pc/make_sorting_contexts.py`, `run_ood_benchmark.sh` | 일회성 스크립트. 추가 태스크용 context 생성이 다시 필요하면 `ood/`나 `preprocess/`로 옮긴다 |
| `feature/method4`의 `fm_l2c2/`와 `train.py` extension hook | 결정 1. merge하지 않음 |
| `lipo.py`의 blend 경로, `--postprocess blend` | 결정 2 |
| `experiments/lipo_*_20261104/` | blend LiPo 기준의 일회성 비교 보고. 삭제(보고서 본문 보존 여부는 확인 후) |
| `codex/` (QP123.md, RTC_QP123.md, `run_qp123*.py`) | suite runner로 대체. 문서 내용은 README에 흡수 |
| `docs/inserting_handoff_20261107.*`, PDF와 `:Zone.Identifier` | 작업 메모와 다운로드 부산물. `.gitignore`에 `*:Zone.Identifier` 추가 |

**유지**: `fm_baseline/models`, `fm_baseline/{train,sample,eval_*,lipo,rtc,smoothness}.py`, `fm_constraint/` 핵심 파일, `configs/`, `preprocess/`(추가 태스크 포함), `ood/`(3.1절로 통합), `visualizer/`, 각 방법·OOD의 단위 테스트.

### 4단계 결과 (2026-11-08, 브랜치 `integrate/phase4-cleanup`)
- **⑤ 하나로 통일**: `fm_baseline/lipo.py`의 blend 경로와 자체 QP(`minimum_jerk`)를 삭제하고 `fm_constraint/lipo.jerk_qp`를 호출하게 했다.
  - baseline 평가는 `--postprocess none|lipo`, `--lipo-path-bound-m`만 받는다.
  - `run_lipo_comparison.py`, `run_lipo_sharded.py`는 none 대 lipo만 비교한다.
  - 합성 fixture의 목표 jerk RMS는 78.7 → 41.4 m/s³(약 47% 감소)다. 이전 blend+QP 정의는 21.7이었다. 테스트 기준을 50%에서 40% 감소로 바꿨다.
- **삭제**:
  - `experiments/`(blend 기준 2026-11-04 LiPo 보고 3개)
  - `fm_constraint/results/legacy/`, `order_projection_pc_vlsa_validation.json`, `integration_validation.json`, `integration_source_manifest.json`, `method123_runner_validation.json`
  - `docs/inserting_handoff_20261107.*`
  - `requirements-lipo.txt`(clarabel은 `requirements.txt`에 있다)
  - 참고 논문 PDF는 추적만 해제했다(`git rm --cached`, `docs/*.pdf` ignore). 로컬 파일은 그대로다.
- **계획과 다르게 유지**: `fm_constraint/tests/test_local_models.py`. HardFlow 없이 동작하는 모델 수식·Euler·checkpoint 로드·외부 import 차단 테스트이므로 남겼다.
- **이미 앞 단계에서 처리**: `fm_pc/`, ACG, PC 주석·호환 코드, GPU 분할 스크립트, `codex/`(master에 없음), L2C2(merge 안 함).
- **검증**:
  - pytest **151 passed**. visualizer 테스트를 처음 실행했다(scratchpad 가상환경에 `visualizer/requirements.txt`를 설치).
  - D3IL `fm_baseline/eval_{pushing,inserting}.py --postprocess lipo` smoke 통과.
- **결정 대기**: `visualizer/samples/`의 내장 대시보드는 2026-11-04 blend 정의의 Baseline/Blending/LiPo 기록이다. 재생은 그대로 되지만 현재 ⑤와 정의가 다르다(12절).

---

## 6. 5단계: 문서 정리
- `README.md`: 설치(HardFlow 없음), 5개 방법 표, 실행 예시, suite runner를 정리한다.
- `fm_constraint/README.md`: schema v4와 실행 순서(4.2)를 반영한다.
- `docs/`에는 방법별 명세(Projection, PF2MP, VLSA, RTC, LiPo), `report.md`, research overview만 남긴다.

---

### 5단계 결과 (2026-11-08, 브랜치 `integrate/phase5-docs`)
- **visualizer 샘플 삭제** (12절 결정 (b)):
  - 삭제: `visualizer/samples/`, 시작 시 샘플을 store로 복사하던 코드
  - 대시보드 기본 제목: "Baseline / Blending / LiPo" → "비교"
  - 테스트: 빈 store에서 대시보드가 비어 있음을 확인하도록 바꿨다.
- **새 문서**: `docs/METHODS.md`. 현재 코드 기준으로 5개 방법과 제약 집합, 실행 순서, 상호작용 규칙, 조합 ID와 비교 쌍, 출처를 정리했다.
- **삭제한 문서**: `docs/FM_Constraint_Integral.md`, `Predict_Correct_d3il.md`, `Projection_d3il.md`, `VLSA_d3il_exp.md`. PC 시대의 통합 명세이거나, 폐기한 제약·코드(`fm_pc`, `fm_projection`, `vlsa.py`) 기준이었다. git 이력에 남아 있다.
- **고친 문서**:
  - 루트 `README.md`: 문서 표, 저장소 구조, 8절 5개 방법, 10절 테스트·viewer
  - `docs/README.md` 3절: METHODS로 연결
  - `docs/PF2MP.md`: 파이프라인 위치, CLI, schema (이후 사용자가 문서를 삭제했다)
  - `docs/YFLOW_research_overview.md`: 현재 5개 방법과 다르다는 안내
  - `fm_baseline/README.md` 9·10절, `fm_baseline/LIPO.md`(현재 ⑤ 정의로 재작성)
  - `visualizer/README.md`, `TEAM_GUIDE.md`
- 남은 오래된 용어(VLSA, blend, method123)는 이력을 설명하는 문장에만 남아 있다.

## 7. 6단계: master 반영과 브랜치 정리 (네트워크 작업은 사용자가 실행)

1. Claude: `integrate/all-methods`에서 전체 테스트와 smoke가 통과한 것을 확인한다.
2. 사용자: `git push -u origin integrate/all-methods` → GitHub에서 PR을 만들고 master에 merge한다.
3. 사용자: 다음 원격 브랜치를 삭제한다. `git push origin --delete feature/method1 feature/method2 feature/method3 feature/method123_merge feature/rtc codex/rtc-qp-method123 feature/method4 visualizer feature/ood-benchmark` (`feature/method4`는 결정 1에 따라 merge 없이 삭제)
   - 삭제 전에 Claude가 각 브랜치의 커밋이 master에 포함됐는지(`git branch -r --merged`, 또는 `-s ours` 기록 여부)를 표로 확인해 준다.
4. Claude: 로컬 브랜치를 정리하고 `master`를 fast-forward한다.

---

## 8. 실행 순서 요약

| 순서 | 작업 | 완료 조건 |
|---|---|---|
| 1 | master merge + HardFlow 제거 | hardflow grep 0건, 테스트 통과, all-off 일치 |
| 2 | fm_pc → fm_constraint 이관, OOD 두 구현 통합(3.1) | `fm_pc/` 없이 8개 태스크 OOD reset smoke 통과, 두 구현자 확인 |
| 3 | ④ RTC, ⑤ LiPo 연결, schema v4 | 4.4의 1~5 통과 |
| 4 | legacy 삭제 | 테스트 통과, 삭제 목록 commit |
| 5 | 문서 | README 실행 예시가 실제로 동작 |
| 6 | PR과 브랜치 정리 | 사용자 push, PR merge |

---

## 9. 결정 사항 (2026-11-07 확정)

| # | 항목 | 결정 | 반영 위치 |
|---|---|---|---|
| 1 | ACG(`fm_pc/constraints/acg.py`, `check_acg.py`, `build.py`의 `--acg-*`, `inject.py`의 `field=`), L2C2(`feature/method4`, `fm_l2c2/`, `train.py`의 extension hook) | **삭제**. `feature/method4`는 merge하지 않고 브랜치만 삭제 | 3절, 5절 |
| 2 | ⑤ LiPo의 정의 | **codex RTC 브랜치의 고정 jerk QP**(`lipo.py` `mode="qp"`, 보정 한도 0.3 mm, `frozen_prefix`)로 한다. master의 blend 방식(`blend_steps`, `blend_bound_m`)은 삭제 | 4.1, 4.3 |
| 3 | 추가 태스크(avoiding/aligning/sorting 2·4·6/stacking), `preprocess/prepare_task.py` | **유지** | 3절 |
| 4 | terminal-plan VLSA | **삭제**. ③ = 실행 직전 H=1 QP(`methods.filter`, 원래 VLSA의 의미)로 확정. ①과 ③이 둘 다 ON이면 둘 다 실행한다(자동 off 없음). 계획 끝 1회 투영은 ① `mode: final`로 대체 | 10절, 4.1 |
| 5 | 예전 설정·결과 호환(schema v2/v3, `hardflow_commit` provenance, PC 거부 분기) | **지원하지 않음**. v4만 지원 | 2절, 5절 |
| 6 | OOD | **유지**. 두 사람이 따로 만든 구현을 **하나로 합친다** | 3.1절 |
| 6′ | `experiments/lipo_*_20261104/` | blend LiPo 기준의 일회성 보고이므로 삭제 대상. 보고서 본문만 남길지는 5절에서 확인 | 5절 |

---

## 10. terminal-plan VLSA (2026-11-07 결정: 삭제, ③ = 실행 직전 필터)

### 정의
- **terminal-plan VLSA**(현재 `methods.vlsa`): Predict의 **마지막 Euler 스텝까지 끝난 최종 샘플**, 즉 H=16개 행동 전체를 계획 시점의 관측(블록 위치, 직전 실행 행동)으로 만든 제약 집합에 **한 번의 QP로 공동 투영**한다. 계획당 1회 실행된다. 가속도 제약이 인접 행동을 묶기 때문에 16개를 함께 푼다.
- **③ Executed-action filter**(`execution_filter`, `vlsa.execution_recheck`): 큐에서 **실제로 실행할 행동 하나**만, 실행 직전의 **최신 관측**으로 H=1 QP를 푼다. 매 제어 step마다 실행된다. 행동이 수정되면 남은 큐를 버리고 재계획한다.

### 쟁점
1. "원래 모델이 마지막 샘플링으로 생성한 결과에 적용"한다는 이해는 **terminal-plan VLSA에 대해서는 맞다**. 다만 이 저장소에서 VLSA를 처음 설계할 때(`docs/VLSA_d3il_exp.md` 0.3절)의 정의는 **실행할 각 행동마다 푸는 실행 직전 QP**였다. 즉 원래 VLSA는 ③에 해당한다. terminal-plan은 2026-10-24 통합 때 생긴 변형이다.
2. terminal-plan VLSA는 **① Projection `mode: final`과 같은 연산**이다. 같은 제약, 같은 QP, 같은 시점에 적용된다. 그래서 현재 코드도 `Projection final + VLSA` 조합을 거부한다(`fm_constraint/README.md`). ① `mode: every`를 켠 상태라면 terminal VLSA는 수치 오차 수준의 중복이다.
3. ③과의 차이: terminal은 계획 시점의 블록 위치로 고정한 제약이고 미래 행동 간 가속도까지 고려한다. ③은 최신 관측을 쓰지만 한 step만 본다.

### 경위 (사용자 설명, 2026-11-07)
terminal 위치는 PC Correct 때문에 생겼다. PC는 Euler가 끝난 뒤 계획을 여러 번 다시 수정했으므로, VLSA를 PC 뒤로 보내야 했다. PC를 PF2MP로 대체하면서 PF2MP가 Predict 안으로 들어갔고, Euler 이후에 계획을 수정하는 단계가 없어졌다. 그 결과 terminal VLSA는 Projection과 겹치게 됐다.

### 결정: 자동 off 대신 "③ = 실행 직전 필터"로 되돌리기
- ① every 마지막 Euler 스텝(`k = N-1`)에서도 투영된다. 따라서 ①이 켜진 상태에서 terminal VLSA는 같은 해를 한 번 더 계산할 뿐이다(멱등).
- 외부에는 ON으로 표시하고 내부에서 자동으로 끄면, 결과표의 ③ ON과 ③ OFF 행이 실제로는 같은 실험이 된다. 그러면 Complementarity Δ(①+③ vs ①)가 구조적으로 0이 되고, 그 이유가 metadata에만 남는다. 재현성과 보고의 정직성 면에서 피한다.
- ③을 원래 VLSA의 의미인 **실행 직전 H=1 QP**로 정의하면, ①(계획 시점 기하, H개 공동 투영)과 ③(실행 시점 최신 블록 위치와 실제 직전 행동)은 서로 다른 단계가 된다. 둘 다 켜졌다고 표시하면 실제로 둘 다 실행된다.
- terminal 모드는 삭제한다. "계획 끝에서 한 번만 투영"이 필요하면 ① `mode: final`을 쓴다.
- 기록: ③의 개입 횟수와 수정량(`execution_filter.intervention_norm_m`)을 rollout마다 저장한다. ①이 켜져 있을 때 ③이 실제로 얼마나 개입하는지가 곧 중복도 측정값이 된다.

### 적용 사항
- `methods.vlsa`(terminal)를 **삭제**하고, 필요하면 ① Projection의 `mode: final`을 ablation으로 쓴다.
- ③을 이름까지 `methods.filter`(= 원래 의미의 VLSA)로 통일한다.
- codex 브랜치의 기존 RTC 실험(③을 terminal VLSA로 실행한 결과)은 ③ 결과가 아니라 "① final 변형" 결과로 표시한다.

---

## 11. 남은 논의 (1단계 시작과는 무관)

1. **⑤ LiPo 이후의 제약 처리**: LiPo jerk QP는 ①로 투영된 계획을 최대 0.3 mm까지 다시 움직이므로 ① 제약을 다시 위반할 수 있다. 현재 규칙대로라면 ①이 ON일 때 최종 검사에서 계획 전체를 거부하고 episode가 실패한다. 선택지는 세 가지다.
   - (a) ①이 ON이면 ① 제약을 LiPo QP에 함께 넣는다(jerk 최소화 + 제약 동시 만족).
   - (b) ⑤ 이후에는 검사하지 않고, 위반은 측정만 하며 ③(ON일 때)이 실행 시점에 처리한다.
   - (c) ⑤를 ① 앞에 둔다. 다만 ①은 Predict 내부 단계라 순서상 불가능하고, `mode: final`일 때만 가능하다.
   - **결정 (2026-11-07): (a)**. ①과 ⑤가 함께 ON이면 LiPo QP에 ①의 선형 제약 `G·(계획+ε) ≤ d`를 추가해 한 번에 푼다. ① 투영 후의 계획은 ε=0으로 이미 해가 되므로 항상 해가 존재한다. ①이 OFF면 LiPo만 단독으로 실행한다. solver는 OSQP 하나로 통일한다.
2. **구현 1의 추가 섭동** — **결정 (2026-11-07): 유지**. 조건군 D(실행 중 push: `seam`/`mid`)와 E(inference delay d step, 시작 관측을 d step 전으로 하고 새 chunk는 index d부터 실행)를 A/B/C 14개 조건에 추가한다. seed는 구현 2의 해시 규칙으로 맞춘다. D는 ③의 효과(최신 관측)를, E는 ④ RTC의 `delay_steps > 0` 효과를 측정하는 조건이다. 강도 단계(push 거리, delay d)는 2단계에서 정한다.
3. **추가 태스크** (2026-11-07 결정): 실험 여부가 아직 미정이다. 이번 통합은 **Pushing·Inserting에만 집중**한다. 추가 태스크의 코드(`tasks.py`, `preprocess/prepare_task.py`)는 남기되, OOD 재생성·검증·5개 방법 연결은 Pushing·Inserting만 수행한다. checkpoint도 이 두 태스크만 있다.
4. **진행 중인 실험** (2026-11-07 확인): 진행 중인 실험은 없다고 간주한다. 팀원마다 다른 실험 조건 방식을 통일하는 것이 이 통합의 목적이다. 실험 환경 통일의 세부 사항은 브랜치 통합 후에 사용자가 따로 전달한다.
5. **PR 단위** — **결정 (2026-11-07): 단계별 PR**. 각 단계는 테스트가 통과하는 상태로 끝내고, 브랜치 `integrate/phase<N>-*`를 master 기준으로 만든다. push와 PR 생성은 사용자가 실행한다.

---

## 12. 4단계 이후 결정
1. **visualizer 내장 샘플**: 현재 6개 샘플(`visualizer/samples/`)은 이전 blend LiPo 기록이다. 선택지는 세 가지다.
   - (a) 이력 데모로 유지하고 README에 표시한다.
   - (b) 삭제한다.
   - (c) 현재 ⑤로 다시 기록한다(D3IL 재실행과 관절 기록 필요).
   - **결정 (2026-11-08): (b) 삭제.** 5단계에서 처리했다.

