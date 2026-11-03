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
