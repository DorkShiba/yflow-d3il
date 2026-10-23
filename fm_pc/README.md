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
