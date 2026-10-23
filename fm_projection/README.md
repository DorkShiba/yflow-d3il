# 방법 ①: 매 ODE 스텝 투영 (`fm_projection/`)

meeting_1008 10쪽. 베이스라인 flow matching 정책의 Euler 샘플링에서 매 스텝 뒤에 가능 집합 $\mathcal C$로 투영한다.

$$x_{k+1} = \Pi_{\mathcal C}\big(x_k + \Delta t\, v_\theta(x_k, t_k)\big), \qquad \Pi_{\mathcal C}(y) = \arg\min_{z\in\mathcal C} \lVert z - y \rVert^2$$

학습이 필요 없는 샘플링 시점 방법이다. `fm_baseline/`은 수정하지 않고 `train.py`·`contexts/`만 가져다 쓴다. 평가 스크립트는 `fm_baseline/eval_*.py`의 루프를 그대로 복사하고 정책 생성 뒤에 래퍼만 끼운 것이라, `--inject none`이면 베이스라인과 같은 객체·같은 결과다.

| 파일 | 내용 |
|---|---|
| `constraints/sets.py`, `qp.py` | 가능 집합과 정확한 박스 QP (ADMM $\rho=0.3$, CPU float64) |
| `constraints/inject.py` | `ProjectionInjector`, 훅이 있는 Euler 루프 (꺼지면 HardFlow solver와 비트 단위 동일) |
| `constraints/build.py` | CLI 옵션 → 집합·래퍼 |
| `eval_pushing.py`, `eval_inserting.py` | 평가 (베이스라인 루프 + 래퍼) |
| `check_constraints.py` | 시뮬레이터 없는 점검: [1] 꺼짐 == 베이스라인, [2] 투영 정확도, [3] 계획 위반·데모 RMSE |
| `compute_metrics.py` | 성공률·엔트로피·점수, context bootstrap CI, 폭주·행동 한도·jerk |
| `render_rollouts.py` | 저장된 rollout 재생 영상 (`apt-get install -y libosmesa6`, 또는 `--map-only`) |

## 가능 집합 (`--constraints`)

| 이름 | 집합 | 기본값 |
|---|---|---|
| `action_box` | $\lvert a_{t,i}\rvert \le a_{\max}$ | 학습 행동 범위 0.00833 m |
| `tracking` | $\lVert d_0 + \sum_{k\le t} a_k - c_0 \rVert_\infty \le d$ | $d$ = 0.10 m |
| `workspace` | $\mathrm{lo} \le d_0 + \sum_{k\le t} a_k \le \mathrm{hi}$ | 학습 데이터의 목표 손 범위 |
| `step_norm` | $\lVert a_t \rVert_2 \le r$ | $r = \sqrt2\, a_{\max}$ |

$d_t$는 목표 손 위치, $c_0$는 계획 시점의 실제 손 위치. 기본값은 실행 폴더의 `normalizer.npz`에서 채운다. 박스형 집합의 교집합은 QP 하나로 정확히 풀고(SLSQP 대비 1e-10), `step_norm`이 섞일 때만 Dykstra를 쓴다.

## 실행 (레포 루트에서)

```bash
R=fm_baseline/results/pushing_baseline
python fm_projection/check_constraints.py --run-dir $R --constraints action_box,tracking --n-obs 32
python fm_projection/eval_pushing.py --run-dir $R --ckpt baseline_ema.pt --n-traj 4 --execute-steps 8 --tag v100 \
    --inject every --constraints action_box,tracking          # --inject final: 마지막에 한 번만
python fm_projection/compute_metrics.py --eval-dir $R/eval_pushing_test_v100_<base> --label baseline \
    --eval-dir $R/eval_pushing_test_v100_every_<stamp> --label every
```

평가 결과는 베이스라인과 같이 `--run-dir` 아래 `eval_pushing_<contexts>_<tag>_<stamp>/`에 쓰인다(gitignore).

## Pushing 결과 (V100, test 30 context × 4, 2026-10-22)

`results/pushing_v100_20261022/`

| | 성공률 | 베이스라인 대비 | 엔트로피 | 점수 | 행동 한도 초과 스텝 |
|---|---|---|---|---|---|
| 베이스라인 | 91.7 | – | 0.509 | 0.713 | 16.8% |
| final | 92.5 | +0.8 [−5.8, +6.7] | 0.552 | 0.739 | 0% |
| every | 91.7 | +0.0 [−5.8, +5.8] | 0.535 | 0.726 | 0% |

위반은 0이 되지만 성공률 차이는 없다. 이 $\mathcal C$는 학습 범위에서 나와 베이스라인이 거의 지키고, 실패의 원인과 무관하다(폭주 120회 중 2회). 베이스라인이 실제로 어기는 제약(속도 제한, 금지 구역, Inserting 벽)에서 다시 비교해야 한다.
