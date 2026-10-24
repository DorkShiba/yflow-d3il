# Projection: D3IL 제약과 적용 방식

> 이 문서는 통합 전 구현의 기록이다. 현재 실행 코드는 `fm_constraint/`로 이관했으며, 이전 소스는 `fm_constraint/results/integration_source_manifest.json`에 기록한 Git 커밋에서 확인할 수 있다. 현재 설정과 실행 방법은 [통합 README](../fm_constraint/README.md)를 따른다.

이 문서는 현재 `fm_projection/` 구현을 기준으로 한다. 기존 FM 모델·학습 손실을 유지하고, **Euler 샘플링 도중 생성한 계획을 가능 집합으로 투영**한다.

`--inject every`에서는 노이즈가 남아 있는 중간 샘플에도 제약을 적용한다는 이해가 맞다. 정확히는 **매 Euler 갱신 직후 샘플의 행동 채널을 물리 단위로 변환해 투영**하고, 그 결과를 다음 신경망 입력으로 사용한다. 중간 상태에 맞춰 제약을 노이즈화하거나 별도 노이즈를 추가하는 처리는 없다. `--inject final`은 생성 완료 후 한 번만 투영하는 비교 모드다.

## 1. 변수와 적용 대상

생성 시간 `τ∈[0,1]`과 계획의 제어 시점 `h=0,…,H−1`를 구분한다. 기본 horizon은 `H=16`이다. 모델의 계획은 `[행동, 관측]` 순서로 Pushing `(B,H,14)`, Inserting `(B,H,18)`이며 앞 2개 채널이 목표 손 XY 증분 `a_h`다. 단위는 m/제어 스텝이다.

현재 관측에서 목표 손 위치를 `d₀`, 실제 손 위치를 `c₀`라 두고, 실행 후 목표 위치는 다음처럼 재구성한다.

\[
d_h^+=d_0+\sum_{j=0}^{h}a_j=d_0+(La)_h,
\qquad L_{hj}=\mathbf{1}[j\le h].
\]

**투영이 직접 수정하는 것은 행동 채널뿐이다.** 모델이 미래 관측 채널에 생성한 손·블록 위치를 투영하지 않는다. 목표 손 위치 제약도 미래 관측 예측값 대신 위 행동 누적식을 사용한다. 첫 행 관측은 실제 현재 관측으로 고정하고, 매 Euler 스텝 및 최종 처리 뒤에 다시 조건을 적용한다.

## 2. 제약 집합

`fm_projection/constraints/sets.py`에 다음 집합이 정의되어 있으며, 선택한 집합의 교집합을 사용한다. Pushing과 Inserting에 같은 형태를 적용한다.

| CLI 이름 | 제약식: 모든 계획 시점에 적용 | 제한하는 값 |
|---|---|---|
| `action_box` | `−a_max ≤ a_{h,i} ≤ a_max` | 축별 목표 손 증분 |
| `tracking` | `‖d_h^+−c₀‖∞ ≤ track_d` | 계획 시작 시 실제 손과 누적 목표 사이의 거리 |
| `workspace` | `ws_lo ≤ d_h^+ ≤ ws_hi` | 누적 목표의 XY 사각형 영역 |
| `step_norm` | `‖a_h‖₂ ≤ r` | 목표 손 한 스텝 증분의 L2 크기 |

`tracking`은 한 계획 내에서 고정된 `c₀`를 사용한다. 미래 실제 손 위치와 목표의 차이를 예측하는 제약이 아니다. `step_norm`도 실제 손의 이동량이나 `a_h−a_{h−1}`의 변화량 제한은 아니다.

### 기본 파라미터

`fm_projection/constraints/build.py`는 해당 run의 `normalizer.npz`에서 `a_max=max(scale[:2])`, `r=√2·a_max`를 계산한다. `workspace`는 목표 손 채널의 `center±scale`이다. 현재 저장된 baseline 통계의 값은 다음과 같다. 숫자는 반올림했다.

| 파라미터 | Pushing | Inserting |
|---|---|---|
| `a_max` | 약 `0.008333461 m` | 약 `0.008333461 m` |
| `r` | 약 `0.011785294 m` | 약 `0.011785294 m` |
| `track_d` | `0.10 m` | `0.10 m` |
| `ws_lo` | `[0.2654675, −0.2964039] m` | `[0.0414766, −0.3392341] m` |
| `ws_hi` | `[0.7797633, 0.3949761] m` | `[0.8012692, 0.6660334] m` |

`action_box`와 기본 `step_norm`을 함께 쓰면 `‖a_h‖₂≤√2·a_max`가 이미 축별 box에서 따라오므로 기본 norm 제약은 추가 제한을 주지 않는다. 더 작은 `r`를 지정하면 별도 제한이 된다.

`workspace`는 학습 목표 위치의 범위다. Inserting의 내부 장벽, 손·블록 충돌, 전체 로봇 링크, 블록의 목표 도달 조건은 이 집합에 포함되지 않는다. [`fm_constraint/`](../fm_constraint/)의 YAML 기반 실행 직전 필터와도 별도 구현이다.

## 3. 노이즈가 있는 중간 샘플에 적용하는 순서

`fm_projection/constraints/inject.py`의 처리 순서는 다음과 같다.

1. 현재 관측을 정규화하고 표준정규 노이즈 `X₀`를 만든다. 첫 행의 관측 채널을 현재 관측으로 고정한다. **초기 노이즈 자체에는 사전 투영하지 않는다.**
2. 현재 샘플에서 `vθ(X_k,τ_k)`를 평가한다. 첫 행 관측의 벡터장 성분은 0으로 만든다.
3. `Y = X_k + Δτ·vθ(X_k,τ_k)`로 Euler 갱신한다.
4. `every` 모드이면 `Y`를 역정규화하고, 그 행동 채널을 가능 집합으로 투영한 뒤 재정규화한다.
5. 첫 행 관측을 다시 고정하고 다음 스텝으로 넘긴다. 이때 신경망은 투영된 행동과 기존 관측 예측을 함께 입력받는다.
6. 모든 스텝이 끝나면 `final` 모드에서만 한 번 투영한다. 최종 역정규화된 계획을 반환한다.

정규화 변환을 `R(X)=X⊙scale+center`라 쓰면 `every`의 갱신은 다음과 같다.

\[
Y_k=X_k+\Delta\tau\,v_\theta(X_k,\tau_k),\qquad
X_{k+1}=R^{-1}\!\left(\Pi_{\mathcal C}(R(Y_k))\right).
\]

`Π`의 최소 거리 기준은 **raw 행동의 유클리드 거리**이며 정규화된 전체 상태·행동 거리와 구분한다. 관측 조건 복원은 위 식 이후 적용한다.

### 적용 모드

| 설정 | 적용 시점 |
|---|---|
| `--inject none` | 원래 baseline 정책 사용, 투영 없음; CLI 기본값 |
| `--inject every --skip-first 0` | 첫 Euler 갱신 직후부터 매 스텝 투영 |
| `--inject every --skip-first K` | 인덱스 `k<K`인 초기 K회 갱신은 건너뛰고 이후 투영 |
| `--inject final` | 전체 Euler 적분 후 한 번만 투영; `skip-first`는 영향 없음 |

기본 ODE 10스텝에서 `every, skip-first=0`은 신경망 10회·투영 10회, `final`은 신경망 10회·투영 1회다. `every`에서 `skip-first≥10`이면 투영이 한 번도 실행되지 않으며 마지막 강제 투영도 없다.

중간 샘플에는 아직 물리적으로 타당하지 않은 값이 있을 수 있지만, 구현은 **생성 시간에 무관한 동일한 물리 제약 집합**을 적용한다. 따라서 중간 노이즈 계획의 행동도 최종 행동과 같은 box·누적 위치 제한으로 당겨진다. terminal 추정치를 별도로 복원해 그것만 보정하는 방식은 아니다.

## 4. 가능 집합으로 투영하는 계산

### 4.1. Box 제약: 전체 horizon을 하나의 QP로 처리

행동 box와 누적 위치 box를 합쳐 다음 문제를 푼다. `a_nom`은 해당 Euler 갱신 직후의 raw 행동 계획이다.

\[
\begin{aligned}
a^*&=\arg\min_a\frac12\sum_{h=0}^{H-1}\|a_h-a_{nom,h}\|_2^2\\
\text{s.t. }&a_{lo}\le a\le a_{hi},\\
&s_{lo}\le La\le s_{hi}.
\end{aligned}
\]

`workspace`의 누적 경계는 `ws_lo−d₀`, `ws_hi−d₀`이며, `tracking`은 `c₀−track_d−d₀`, `c₀+track_d−d₀`다. 여러 위치 box는 하한의 최댓값·상한의 최솟값으로 합친다. 누적 제약이 있으므로 일반적으로 한 행동만 clip하는 것으로 끝나지 않고, 계획 전체의 행동을 함께 수정한다.

`fm_projection/constraints/sets.py`이 box 제약들을 QP 하나로 구성하고 `fm_projection/constraints/qp.py`가 계산한다.

- 행동 box만 있으면 `clip(a_nom,a_lo,a_hi)`가 해다.
- 누적 제약이 있으면 `A=[I;L]`에서 필요한 블록을 골라 `z=Aa`로 분리한 ADMM을 사용한다. XY축·배치는 독립적으로 계산한다.
- 계산은 NumPy CPU `float64`, ADMM `rho=0.3`, 최대 5,000회, primal/dual 중단 기준 `1e−11`이다.
- 행동 box가 있으면 반환 직전에 행동을 clip한다. 누적 경계 잔차는 반복 수·허용 오차 및 모델 dtype으로의 복귀 영향을 받는다. 최대 반복 도달 시 예외를 발생시키는 코드는 없다.

### 4.2. 경계 완화

원래 위치 box들이 서로 겹치지 않아 `s_lo>s_hi`가 되면 각 원소의 중간값으로 양쪽 경계를 맞춘다. 행동 box가 함께 있으면 누적 위치 경계를 `[cumsum(a_lo), cumsum(a_hi)]` 안으로 clip한다.

예를 들어 현재 목표가 workspace 밖에 있어 제한된 행동으로 한 스텝 안에 복귀할 수 없다면, 첫 스텝부터 원래 workspace를 강제하는 대신 도달 가능한 경계로 바꿔 복귀를 유도한다. **이때 실제로 푸는 집합은 원래 제약 집합과 달라진다.** 시점별 도달 범위 clip이 임의의 모든 제약 조합에 대해 공동 feasible 여부를 증명하는 것은 아니다.

`violation()`은 원래 제약의 위반을, `BoxStack.qp_residual()`은 완화 후 QP 경계의 잔차를 측정한다. 둘을 구분해야 한다. 투영을 했다는 사실만으로 모든 상황의 원래 제약 만족을 주장할 수 없다.

### 4.3. `step_norm`: 원판 투영과 Dykstra

`step_norm` 하나만 쓰면 각 행동을 다음처럼 반지름 `r`의 원판으로 축소한다.

\[
a_h^*=a_{nom,h}\min\!\left(1,\frac{r}{\|a_{nom,h}\|_2}\right).
\]

box들과 함께 쓰면 box 전체를 하나의 `BoxStack`으로 묶고, 원판과의 교집합 투영은 **보정 변수를 유지하는 Dykstra 반복**으로 구한다. 기본 `dykstra_iters=200`이며 조기 수렴 판정 없이 정해진 횟수를 반복한다. 따라서 유한 반복 결과의 정확도는 확인이 필요하며, 단순 box QP와 동일한 정확도 보장을 붙이면 안 된다.

## 5. 실행과 해석

```bash
# 중간 샘플마다 제약 적용
python fm_projection/eval_pushing.py \
  --run-dir fm_baseline/results/pushing_baseline \
  --ckpt baseline_ema.pt --inject every \
  --constraints action_box,tracking \
  --execute-steps 8

# 생성 완료 후 한 번만 제약 적용
python fm_projection/eval_pushing.py \
  --run-dir fm_baseline/results/pushing_baseline \
  --ckpt baseline_ema.pt --inject final \
  --constraints action_box,tracking \
  --execute-steps 8
```

제약 기본 목록은 `action_box` 하나다. `--constraints`만 지정하고 `--inject`를 생략하면 투영이 켜지지 않는다. 수치는 `--constraint-params '{"a_max": 0.008, "track_d": 0.08, "r": 0.01}'`처럼 덮어쓴다. Inserting은 `eval_inserting.py`와 해당 run을 사용한다.

평가 루프는 계획의 앞 `execute_steps`개 행동을 큐에 넣고 `des ← des+a`로 실행한 뒤 재계획한다. **큐의 각 행동 실행 직전에는 투영을 다시 수행하지 않는다.** 따라서 `tracking`의 기준 실제 손은 계획 시점의 `c₀`다. Pushing의 실행 묶음 기본값은 1, Inserting은 8이므로 예시에서는 8을 명시했다.

## 6. Predict-then-Correct와의 차이

| 항목 | Projection `every` | Predict-then-Correct 기본값 |
|---|---|---|
| 제약 시작 시점 | 첫 Euler 갱신 직후, 노이즈가 남은 중간 계획 | 무제약 terminal 초안 예측 이후 |
| 보정 대상 | 계획의 행동 값 자체 | 보정 루프의 행동 벡터장 |
| 계산 | raw 행동을 가능 집합으로 최소 거리 투영 | 선형 장벽의 CBF 부등식 아래 명목 벡터장 최소 수정 |
| 시간 진행 | 원래 생성 시간 0→1 | 예측 0→1 후 보정 시간을 다시 0→1로 진행 |
| `step_norm` | 지원; box와 결합 시 Dykstra | PC CBF에서는 미지원 |
| 최종 처리 | `every`에서는 마지막 Euler 갱신 뒤에도 투영 | 마지막 CBF Euler 갱신 뒤 별도 투영 없음 |

PC의 장벽식·파라미터·경계 완화는 [Predict-then-Correct 문서](Predict_Correct_d3il.md)에 정리했다. 두 방법 모두 목표 명령 계획을 제한하며 실제 손·블록의 비충돌이나 작업 성공을 직접 보장하지 않는다.
