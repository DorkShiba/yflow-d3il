# 방법 ②: Predict-then-Correct + 유한 시간 CBF (`fm_pc/`)

meeting_1008 12쪽, SafeFlowMatcher(Yang, Jang, Han, ICLR 2026) Algorithm 1. 노이즈에서 계획 초안을 한 번에 예측하고, 그 보정 경로만 CBF-QP로 걸러 가능 집합 $\mathcal C$ 안에 둔다.

$$\text{Predict:}\quad \hat x_1 = x_0 + v_\theta(x_0, 0)$$

$$\text{Correct:}\quad u^\star = \arg\min_u \big\lVert u - \alpha(1-t)\, v_\theta(x_t, t)\big\rVert^2 \quad \text{s.t.}\quad \nabla b_j^\top u + \varepsilon\,\mathrm{sgn}(b_j-\delta)\,\lvert b_j-\delta\rvert^\rho \ge 0 \ \ \forall j$$

학습이 필요 없는 샘플링 시점 방법이다. `fm_baseline/`은 수정하지 않고 `train.py`·`contexts/`만 가져다 쓴다. 평가 스크립트는 `fm_baseline/eval_*.py`의 루프를 그대로 복사하고 정책 생성 뒤에 래퍼만 끼운 것이라, `--sampler euler`(기본)이면 베이스라인과 같은 객체·같은 결과다.

| 파일 | 내용 |
|---|---|
| `constraints/pc.py` | `PCSampler`: 예측 $T^p$스텝, 보정 $T^c$스텝, 스텝마다 CBF-QP |
| `constraints/sets.py`, `qp.py` | 가능 집합(장벽 $b_j$의 면)과 정확한 박스 QP |
| `constraints/inject.py` | 래퍼와 훅이 있는 Euler 루프 (꺼지면 HardFlow solver와 비트 단위 동일) |
| `constraints/build.py` | CLI 옵션 → 집합·샘플러 |
| `eval_pushing.py`, `eval_inserting.py` | 평가 (베이스라인 루프 + 래퍼) |
| `check_constraints.py` | 시뮬레이터 없는 점검: [1] 꺼짐 == 베이스라인, [2] none / pc / pc-noCBF 계획의 위반·데모 RMSE |
| `compute_metrics.py` | 성공률·엔트로피·점수, context bootstrap CI, 폭주·행동 한도·jerk |
| `render_rollouts.py` | 저장된 rollout 재생 영상 (`apt-get install -y libosmesa6`, 또는 `--map-only`) |

장벽 $b_j$는 가능 집합의 각 면을 반폭으로 나눈 값이다. 면이 선형이라 CBF 조건이 "$u$ 상자 × $Lu$ 상자"가 되어 스텝마다 `BoxQP`로 정확히 푼다. slack은 쓰지 않는다.

| `--pc-params` | 기본값 | 의미 |
|---|---|---|
| `tp`, `tc` | 1, ode_steps − tp | 예측·보정 스텝. 신경망 호출 10회로 베이스라인과 같다 |
| `alpha` | 2.0 | 보정 속도장 $\alpha(1-t)v_\theta$의 크기 (논문 Maze2D) |
| `rho`, `delta` | 0.9, 0.01 | 유한 시간 CBF 지수, 정규화 장벽의 여유 (논문 Maze2D) |
| `eps` | 자동 | Euler 한 스텝이 선형 장벽을 넘지 않는 최대값의 0.9배 ($T^c=9$이면 7.07) |

가능 집합(`--constraints`)은 `action_box`($\lvert a_{t,i}\rvert \le$ 학습 행동 범위 0.00833 m), `tracking`($\lVert d_t - c_0\rVert_\infty \le$ 0.10 m), `workspace`, `step_norm` 중에서 고른다. 기본값은 실행 폴더의 `normalizer.npz`에서 채운다.

## 실행 (레포 루트에서)

```bash
R=fm_baseline/results/pushing_baseline
python fm_pc/check_constraints.py --run-dir $R --constraints action_box,tracking --n-obs 32
python fm_pc/eval_pushing.py --run-dir $R --ckpt baseline_ema.pt --n-traj 4 --execute-steps 8 --tag v100 \
    --sampler pc --constraints action_box,tracking            # --pc-params '{"alpha": 1.0}', '{"tp": 5}'
python fm_pc/compute_metrics.py --eval-dir $R/eval_pushing_test_v100_<base> --label baseline \
    --eval-dir $R/eval_pushing_test_v100_pc_<stamp> --label pc
```

평가 결과는 베이스라인과 같이 `--run-dir` 아래 `eval_pushing_<contexts>_<tag>_<stamp>/`에 쓰인다(gitignore). CBF만 끈 대조군(pc-noCBF)은 `--constraint-params`로 집합을 CBF가 걸리지 않을 만큼 넓혀 돌린다(쓴 값은 결과 폴더 `results.json`의 `constraint_params`).

## Pushing 결과 (V100, test 30 context × 4, 2026-10-22)

`results/pushing_v100_20261022/`

| | 성공률 | 베이스라인 대비 | 엔트로피 | 점수 | 행동 한도 초과 스텝 |
|---|---|---|---|---|---|
| 베이스라인 | 91.7 | – | 0.509 | 0.713 | 16.8% |
| pc | 98.3 | +6.7 [+1.7, +11.7] | 0.229 | 0.606 | 0% |
| pc, $\alpha=1$ | 95.8 | +4.2 [−0.8, +9.2] | 0.241 | 0.600 | 0% |
| pc, CBF 없음 | 95.0 | +3.3 [−0.8, +7.5] | 0.264 | 0.607 | 8.6% |

성공률 향상은 유의하지만 엔트로피가 절반 이하로 줄어 D3IL 점수는 낮아진다. CBF 없는 대조군도 엔트로피가 같이 줄었으므로 이는 샘플러(1스텝 예측이 조건부 평균에 가까움) 때문이고 CBF의 효과가 아니다. CBF는 위반을 0으로 만들고 성공률을 +3.3%p 더 올렸지만 이 차이는 아직 유의하지 않다. 다음 후보는 같은 호출 수에서 `tp`를 늘리는 것(`--pc-params '{"tp": 5}'`)이다.

## 시간축 교란 guidance (ACG의 conv 버전, `constraints/acg.py`)

ACG(Park et al., ICRA 2026)는 action 토큰의 self-attention을 단위행렬로 바꿔 "시간축 정보가 섞이지 않는" 속도장 $v_\theta^{IC}$를 만들고 그 반대로 외삽한다. 베이스라인 `TemporalUnet`은 attention이 없고(`attention: false`) 시간축을 섞는 것은 kernel 5인 temporal Conv1d뿐이라, **kernel의 가운데 tap만 남기는 것**(= 1x1 conv)을 그 대응으로 쓴다. channel 방향 계산, time embedding, GroupNorm, residual 경로는 그대로다.

$$v = (1+\lambda)\,v_\theta(x,t) - \lambda\, v_\theta^{IC}(x,t), \qquad t_0 \le t < t_1$$

- HardFlow 코드와 체크포인트는 수정하지 않는다. forward hook이 배치의 **뒤쪽 절반**만 교란하고, `[x, x]`를 한 번에 넣어 $v_\theta$와 $v_\theta^{IC}$를 같이 계산한다(GroupNorm은 샘플별이라 두 절반이 섞이지 않는다).
- `--acg-scale 0`(기본)이면 베이스라인과 비트 단위로 같다. `--sampler pc`와 같이 쓸 수 있다(PC의 predict·correct가 보는 $v_\theta$ 자체를 바꾼다).
- `--acg-mode skip`은 `ResidualTemporalBlock`을 residual 경로로 대체한다(STG-R, CVPR 2025). `final`에는 쓸 수 없다.
- 교란 위치(`--acg-sites`)와 시간 해상도 (Pushing): `down0` 16, `down1` 8, `down2` 4, `mid` 4, `up0` 4, `up1` 8, `final` 16. Down/Upsample conv는 교란하지 않는다.

```bash
R=fm_baseline/results/pushing_baseline
python fm_pc/check_acg.py --run-dir $R --n-obs 128 --n-samples 8                       # 전 site × λ 오프라인 점검
python fm_pc/check_acg.py --run-dir $R --sites "down2;mid" --t-range 0,0.5 --obs-noise 0.3  # 관측을 분포 밖으로
python fm_pc/eval_pushing.py --run-dir $R --ckpt baseline_ema.pt --n-traj 4 --execute-steps 8 --tag acg \
    --acg-scale 1 --acg-sites mid --acg-t-range 0,0.5                                   # (+ --sampler pc ...)
```

### 오프라인 점검 결과 (Pushing, 검증 관측 128개 × noise 8개, 실행 8 step, 2026-11-03)

같은 초기 noise로 짝지어 비교했다. 단위는 $a_{\max}$. `bJerk`는 청크 시작에서 데모 이력 2 step과의 3차 차분이다. 결과 JSON은 `check_acg.py --out`으로 저장한다.

| 조건 | dA | jerk | bJerk | >amax | div (m) |
|---|---|---|---|---|---|
| 베이스라인 | 0.147 | 0.221 | 0.383 | 0.190 | 0.012 |
| $v^{IC}$ 단독, mid | 0.265 | 0.379 | 0.715 | 0.002 | 0.021 |
| ACG mid, λ=1, t∈[0,1) | 0.208 (+41%) | 0.306 (+39%) | 0.499 | 0.798 | 0.012 |
| ACG mid, λ=1, t∈[0,0.5) | 0.146 (−1%) | 0.215 (−3%) | 0.373 (−3%) | 0.236 | 0.010 (−15%) |
| ACG down2, λ=1, t∈[0,0.5) | 0.144 (−2%) | 0.213 (−4%) | 0.358 (−6%) | 0.258 | 0.010 (−15%) |
| ACG final, λ=1, t∈[0,1) | 1.694 | 3.082 | 1.671 | 0.727 | 0.047 |

- **전제는 성립한다.** 모든 site에서 $v^{IC}$ 단독 궤적이 베이스라인보다 더 떨린다.
- **그러나 guidance는 jitter를 줄이지 못한다.** 전 구간·후반($t\ge0.5$)에 걸면 모든 site·λ에서 jerk가 늘고 `>amax`가 크게 오른다(외삽이 action 크기를 데이터 범위 밖으로 키움). 초반($t<0.5$)에만 걸면 jerk가 최대 4~6% 줄지만 샘플 다양성이 15% 줄고 `>amax`가 늘어, 의미 있는 이득으로 보기 어렵다.
- 원인 추정: 데모 action이 축마다 $\{-a_{\max},0,+a_{\max}\}$의 bang-bang이라, 계획 안의 부호 반전은 거의 없고(청크당 0.002회) jerk 대부분은 0↔$a_{\max}$ 전환이다. ACG가 겨냥하는 "사람 데모의 고주파 떨림"이 이 데이터에는 거의 없다.
- **관측을 분포 밖으로 밀면**(정규화 공간 noise σ=0.1, 0.3) 베이스라인의 `bJerk`가 0.383 → 1.012 → 1.411로 2.6~3.7배가 되는 반면 청크 내부 `jerk`는 0.221 → 0.320 → 0.333(+45~50%)이다. OOD에서 늘어나는 jerk는 주로 **청크 경계**에 있고, 이때도 ACG는 효과가 없다(±3% 이내). 단 `bJerk`는 오프라인에서 데모 이력과 비교한 값이라, 폐루프에서는 직전 계획을 이력으로 다시 재야 한다.

## OOD 벤치마크 (`tasks.py`, `ood.py`, `benchmark.py`, `eval_task.py`, `ood_metrics.py`, `run_ood_benchmark.sh`)

정책·체크포인트·실행 방식(16 생성, 앞 8 실행, Euler 10)은 그대로 두고 **테스트 상황만** 데모 분포 밖으로 옮긴다. 한 실행에 조건 하나(`--ood`)만 켠다. `--ood none`이면 기존 평가 루프와 호출 순서까지 같다(OOD 난수는 전용 Generator를 쓰므로 정책의 torch/numpy 난수열을 건드리지 않는다). Pushing에서 `eval_task.py`와 `eval_pushing.py`는 ID·OOD 모두 rollout이 비트 단위로 같다.

| 조건 | 옵션 | 블록이 있는 task | Avoiding (블록 없음) |
|---|---|---|---|
| ID | `--ood none` | 기준 | 기준 |
| OOD-1 초기 이동 | `--ood shift --ood-shift 0.02` | 모든 블록이 데모 초기 범위 밖 d m에서 시작. 가장 가까운 면부터, 다른 블록·손·벽·작업공간과 겹치지 않는 면을 고른다 | 시작 손 위치를 고정 시작점에서 d m (x−, x+, y− 중 무작위) |
| OOD-2 관측 노이즈 | `--ood obsnoise --ood-noise-pos 0.005 --ood-noise-yaw 5` | 재계획마다 정책이 보는 블록 x, y, yaw에 가우시안 노이즈. 실제 상태는 그대로 | 관측한 손 xy에 노이즈 |
| OOD-3 중간 밀기 | `--ood push --ood-push-dist 0.04` | `--ood-push-step` 이후 첫 재계획 직전에 손에 가장 가까운 블록을 d m 순간이동(무작위 방향, 충돌 없는 방향만). `--ood-push-phase mid`면 청크 중간 | 목표 손 위치를 d m 밀기 |
| OOD-4 추론 지연 | `--ood delay --ood-delay 2` | 다음 청크를 d step 전 관측으로 계획하고 새 청크의 d번째 action부터 실행(naive async) | 같음 |

### task별 설정 (`tasks.py`)

| task | obs / act | action | 블록 (OOD-1/3 대상) | context | 기본 평가 크기 | 밀기 step | 주요 지표 |
|---|---|---|---|---|---|---|---|
| pushing | 12 / 2 | Δ 목표 손 xy | 빨강·초록 | D3IL test 60 | 30 × 4 | 80 | success, mode |
| inserting | 16 / 2 | Δ 목표 손 xy | box 1–3 (미로 벽 회피) | `fm_baseline/contexts/inserting_val.pkl` | 전체 × 4 | 300 | success, boxes |
| avoiding | 4 / 2 | Δ 목표 손 xy | 없음 (손) | 고정 시작 (rollout index) | 100 × 1 | 24 | success, 경로 mode 엔트로피(24), score 0.8s+0.2e |
| aligning | 14 / 3 | Δ 목표 손 xyz | push box만 (target은 고정) | D3IL test 60 | 60 × 4 | 80 | success, mode 엔트로피 |
| sorting_2/4/6 | 4+4n / 2 | Δ 목표 손 xy | 빨강 k, 파랑 k (공통 시작 범위, 목표 영역 벽 회피) | `fm_baseline/contexts/sorting_{n}_test_contexts.pkl` | 30 × 4 | 100 / 150 / 250 | success, 순서 mode 엔트로피 |
| stacking | 23 / 8 | Δ 목표 관절 7 + 그리퍼 폭(절대값) | 테이블 위에 있고 손가락 사이가 아닌 블록 | D3IL test 100 | 30 × 4 | 150 | success 1/2/3단, 쌓는 순서 엔트로피(6) |

- 관측 배치는 `preprocess/prepare_task.py`(학습)와 `tasks.TaskSpec.obs`(평가)가 같아야 한다. `python preprocess/check_prepare_task.py`가 시뮬레이터에서 원본 pickle 형식으로 기록한 뒤 두 경로를 비교한다(전 task 오차 < 2e-7 확인).
- Sorting: D3IL의 `get_contexts.py`는 gym Box 공간 난수를 seed하지 않아 test context를 재현할 수 없고 배포도 안 됨. `make_sorting_contexts.py`가 같은 sampler에 seed를 넣어 1200개를 만든다(파일은 레포에 포함). 2/4/6 상자 파일은 의도적으로 같다(앞 n개 슬롯을 사용).
- Stacking의 jerk 지표는 관절 7개 기준(rad, $\Delta t$ = 0.030 s), 그리퍼는 제외.
- 블록 반폭·손 반경은 task별(`block_half`, `hand_r`), 데모 초기 범위는 `INIT_RANGE`에 있다. 적용된 이동·밀기 기록은 `results.json`의 `ood.events`.

rollout에는 기존 키에 더해 `<ep>_seams`(새 청크가 실행을 시작한 step), `<ep>_pobs`(각 청크를 계획한 관측), `<ep>_event`(밀기 step, 없으면 −1)를 저장한다.

### VESSL에서 돌리는 순서

```bash
# 1) 전처리 (task별, raw = D3IL 원본 데모 폴더)
python preprocess/prepare_task.py --task stacking --raw <raw>/stacking/... --d3il /root/d3il-work/d3il \
    --out /root/d3il-work/codex/processed/stacking_sincos_v1
# 2) 학습 (pushing baseline과 같은 설정)
python fm_baseline/train.py --task stacking --steps 1000000 --scheduler cosine --ema-decay 0.995 --save-every 25000 --tag main
# 3) 벤치마크 (ID + 9개 조건 → <run>/ood_logs_<TAG>_<stamp>/summary)
RUN=<run dir> bash fm_pc/run_ood_benchmark.sh stacking
QUICK=1 RUN=<run dir> bash fm_pc/run_ood_benchmark.sh sorting_4     # 빠른 점검 (4 context × 1)
TAG=pc JOBS=2 bash fm_pc/run_ood_benchmark.sh pushing --sampler pc --constraints action_box,tracking
python fm_pc/ood_metrics.py --eval-dir <ID 폴더> --label id --eval-dir <OOD 폴더> --label push5cm
```

split: aligning·stacking은 D3IL 공식 `train/eval_files.pkl`(900/100, 1000/95). avoiding은 공식 split이 없고, sorting 목록은 배포되지 않아서 파일명을 정렬한 뒤 `split_data.py`처럼 `random.seed(42)` shuffle, 앞 10 %를 val로 둔다(사용한 목록은 `manifest.json`).

`ood_metrics.py` 지표 (단위 $a_{\max}$, Δ = 같은 context에서 ID 대비 차이, 95% CI는 context bootstrap):

| 지표 | 정의 |
|---|---|
| `J_seam` | 청크 경계 $s$에서 $\lVert a_s - 2a_{s-1} + a_{s-2}\rVert / a_{\max}$ (목표 위치의 3차 차분) |
| `J_in` | $t, t-1$이 경계가 아닌 step의 같은 값 (청크 내부) |
| `jump` | 경계에서 $\lVert a_s - a_{s-1}\rVert / a_{\max}$ |
| `JerkRMS` | 목표 위치의 $\sqrt{\mathrm{mean}\lVert \Delta^3 d / \Delta t^3\rVert^2}$ (m/s³, Stacking은 rad/s³) |
| `ATV` | $\mathrm{mean}\lVert a_t - a_{t-1}\rVert / a_{\max}$ |
| `J_event` | 밀기 직후 첫 경계의 jerk (OOD-3) |
| `OOD dist` | 계획에 쓴 관측과 가장 가까운 학습 관측 사이 거리(정규화 공간). 조건이 실제로 분포 밖이었는지 확인용 |
