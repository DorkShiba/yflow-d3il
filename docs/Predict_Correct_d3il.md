# Predict-then-Correct: D3IL 제약과 적용 방식

> 이 문서는 통합 전 구현의 기록이다. 현재 실행 코드는 `fm_constraint/`로 이관했으며, 이전 소스는 `fm_constraint/results/integration_source_manifest.json`에 기록한 Git 커밋에서 확인할 수 있다. 현재 설정과 실행 방법은 [통합 README](../fm_constraint/README.md)를 따른다.

이 문서는 현재 `fm_pc/` 구현을 기준으로 한다. 기존 FM 가중치를 그대로 사용하며, 학습 손실을 바꾸지 않고 **샘플링을 예측 단계와 CBF 보정 단계로 나눈다.**

기본값 `tp=1`에서는 노이즈에서 **생성 시간의 terminal sample인 전체 상태·행동 계획**을 신경망 1회 호출로 예측한다. 이후 그 계획을 초기값으로 별도 보정 루프를 수행한다. 여기서 terminal은 로봇 작업의 마지막 상태 하나가 아니라 FM 생성 시간의 끝을 뜻한다. `tp>1`이면 예측도 여러 Euler 스텝으로 수행한다.

## 1. 변수와 제약 대상

- 생성 시간은 `τ∈[0,1]`, 계획 내부의 제어 시점은 `h=0,…,H−1`로 구분한다. 기본 horizon은 `H=16`이다.
- 모델의 입력·출력 계획은 `[행동, 관측]` 순서다. Pushing은 `(B,H,14)`, Inserting은 `(B,H,18)`이며, 앞 2개 채널이 목표 손 XY 증분 `a_h`다. 행동 단위는 m/제어 스텝이다.
- 현재 관측의 목표 손 위치를 `d₀`, 실제 손 위치를 `c₀`라 둔다. 이 값들은 첫 행 관측에 조건으로 고정된다.
- 행동을 실행했을 때의 목표 손 위치는 다음과 같이 계산한다.

\[
d_h^+=d_0+\sum_{j=0}^{h}a_j=d_0+(La)_h,
\qquad L_{hj}=\mathbf{1}[j\le h].
\]

**제약은 행동 채널에 적용한다.** 미래 관측 채널에 모델이 생성한 목표 손 위치를 직접 제한하는 것이 아니라, 현재 목표와 행동 누적으로 `d_h^+`를 재구성한다. CBF-QP는 행동 벡터장만 수정하며, 나머지 관측 채널은 보정 단계의 명목 벡터장을 따른다. 첫 행 관측의 벡터장은 0으로 만들고 매 스텝 현재 관측으로 다시 고정한다.

## 2. 구현된 제약

제약 집합 정의는 `fm_pc/constraints/sets.py`, 기본값과 CLI 연결은 `fm_pc/constraints/build.py`에 있다.

| 이름 | 제약식: 모든 계획 시점에 적용 | 의미 | PC CBF 지원 |
|---|---|---|---|
| `action_box` | `−a_max ≤ a_{h,i} ≤ a_max` | 축별 목표 손 증분 제한 | 지원 |
| `tracking` | `‖d_h^+−c₀‖∞ ≤ track_d` | 계획 시작 시 실제 손 주변에 누적 목표 제한 | 지원 |
| `workspace` | `ws_lo ≤ d_h^+ ≤ ws_hi` | 누적 목표를 고정 XY 사각형 안에 제한 | 지원 |
| `step_norm` | `‖a_h‖₂ ≤ r` | 한 행동의 XY 이동량 제한 | **미지원** |

`PCSampler`는 `kind="a"`인 행동 box와 `kind="s"`인 누적 위치 box만 수집한다. 따라서 CLI가 `step_norm` 이름을 받아도 **다른 box와 함께 지정하면 PC에서 무시되며, 단독 지정하면 오류가 발생한다.** 현재 코드의 PC 비교에는 `action_box`, `tracking`, `workspace`만 사용해야 한다.

### 기본 파라미터

`a_max=max(normalizer.scale[:2])`, `r=√2·a_max`다. `workspace` 경계는 정규화 통계의 목표 손 채널에서 `center−scale`, `center+scale`로 구한다. 현재 저장된 baseline의 `normalizer.npz`를 읽은 값은 아래와 같다. 숫자는 반올림했으며 실행 시에는 해당 run의 통계를 사용한다.

| 파라미터 | Pushing | Inserting |
|---|---|---|
| `a_max` | 약 `0.008333461 m` | 약 `0.008333461 m` |
| `track_d` | `0.10 m` | `0.10 m` |
| `ws_lo` | `[0.2654675, −0.2964039] m` | `[0.0414766, −0.3392341] m` |
| `ws_hi` | `[0.7797633, 0.3949761] m` | `[0.8012692, 0.6660334] m` |

`tracking`의 기준 `c₀`는 한 계획 안에서 고정된다. 미래 실제 손 위치나 매 실행 스텝의 실제 손을 기준으로 한 추종 오차 제약은 아니다. `workspace`는 학습 데이터의 목표 손 범위이며, Inserting 내부 장벽·블록 충돌·로봇 링크 제약은 포함하지 않는다.

## 3. 예측: 제약 없이 terminal 계획 생성

`fm_pc/constraints/pc.py`는 첫 관측을 고정한 정규분포 노이즈 `X₀`에서 시작한다. `Tᵖ=tp`회의 Euler 적분으로 terminal 초안 `X̂₁`을 만든다.

\[
X^p_{k+1}=X^p_k+\frac{1}{T^p}
v_\theta(X^p_k,k/T^p),\qquad \widehat X_1=X^p_{T^p}.
\]

기본 `Tᵖ=1`이면 다음 한 번의 예측이다.

\[
\widehat X_1=X_0+v_\theta(X_0,0).
\]

이 단계에는 CBF나 투영이 없다. terminal 초안은 제약을 위반할 수 있다.

## 4. 보정: CBF-QP로 행동 벡터장 수정

보정 초기값을 `X⁽ᶜ⁾₀=X̂₁`로 두고 **보정 시간 `τ`를 다시 0부터 시작한다.** 예측 단계의 적분을 `τ=1` 이후로 이어가는 방식은 아니다. 보정 횟수는 `Tᶜ=tc`, 보폭은 `Δτ=1/Tᶜ`다.

각 보정 스텝에서 신경망을 다시 평가하고 다음 명목 벡터장을 사용한다.

\[
\widetilde v_\tau=\alpha(1-\tau)v_\theta(X^c_\tau,\tau).
\]

이 중 행동 성분을 물리 단위의 생성 시간 미분 `v_a`로 변환한다. **이 속도는 로봇의 m/s 속도가 아니라 `da/dτ`다.** 정규화 행동 벡터장에 행동 `scale`을 곱하며, 미분에는 `center`를 더하지 않는다.

### 4.1. 선형 장벽

각 box의 제한 대상을 `q=a` 또는 `q=La`라 두고, 하한·상한을 `ℓ,r`, 반폭을 `w=(r−ℓ)/2`라 둔다. 누적 위치 box는 `d₀`를 빼서 `La`의 경계로 변환한다. 각 면의 정규화 장벽은 다음과 같다.

\[
b_{up}=\frac{r-q}{w},\qquad b_{lo}=\frac{q-\ell}{w}.
\]

`b≥0`이면 해당 면의 안쪽이다. 구현은 `w`의 최솟값을 `1e−12`로 둔다. `δ`는 m 단위 margin이 아니라 **반폭으로 정규화한 장벽의 여유**다.

\[
\phi(b)=\varepsilon\,\operatorname{sgn}(b-\delta)|b-\delta|^\rho,
\qquad \nabla_a b^\top u+\phi(b)\ge0.
\]

따라서 행동 box는 `u`에, 누적 위치 box는 `Lu`에 다음 경계를 만든다.

\[
-w\phi(b_{lo})\le u\le w\phi(b_{up}),
\qquad\text{또는}\qquad
-w\phi(b_{lo})\le Lu\le w\phi(b_{up}).
\]

여러 제약의 하한은 최댓값, 상한은 최솟값으로 합친다. 장벽이 음수인 면은 복구 방향의 행동 변화율을 요구한다.

### 4.2. 최소 수정 QP와 Euler 갱신

\[
\begin{aligned}
u^*&=\arg\min_u\frac12\|u-v_a\|_2^2\\
\text{s.t. }&u_{lo}\le u\le u_{hi},\quad
s_{lo}\le Lu\le s_{hi},\\
X^c_{k+1}&=X^c_k+\Delta\tau\,v_{corrected}.
\end{aligned}
\]

`v_corrected`는 명목 벡터장의 행동 성분을 `u* / action_scale`로 바꾼 것이다. 관측 성분은 명목 벡터장을 유지한다. 즉 **계획 자체를 가능 집합으로 즉시 투영하는 대신, 계획을 움직이는 벡터장을 QP로 수정한다.** 마지막 보정 이후 별도의 terminal 투영은 없다.

QP는 `fm_pc/constraints/qp.py`의 `BoxQP`를 사용한다. 축·배치별로 독립 계산하며 NumPy CPU `float64`, ADMM `rho=0.3`, 최대 5,000회, primal/dual 중단 기준 `1e−11`이다. 행동 box만 있으면 단순 clip으로 푼다. 이 ADMM `rho`는 CBF 지수 `rho=0.9`와 별개다.

### 4.3. 충돌하는 경계와 보장 범위

구현은 slack 변수를 두지 않는다. 대신 하한이 상한보다 크면 두 값의 중간으로 맞추고, `u`와 `Lu` 경계가 함께 있으면 `Lu` 경계를 `[cumsum(u_lo), cumsum(u_hi)]` 안으로 clip한다. 이는 행동 경계를 우선하는 **경계 완화**이며, 원래 모든 CBF 부등식을 그대로 푸는 것과 다르다. 시점별 도달 범위 clip만으로 임의의 경계 조합 전체의 공동 feasible 여부가 증명되는 것은 아니다.

초안이 이미 위반 상태일 수 있고 보정은 유한 횟수이므로, 모든 입력·파라미터에서 최종 원래 제약의 만족을 보장한다고 해석하면 안 된다. 원래 집합 위반 여부와 수치 잔차를 별도로 확인해야 한다. 실제 손·블록의 물리적 안전 역시 이 행동 계획 제약만으로 보장되지 않는다.

## 5. 기본 보정 설정과 실행

| 옵션 (`--pc-params`) | 기본값 | 역할 |
|---|---|---|
| `tp` | `1` | 무제약 예측 호출 수 |
| `tc` | `max(ode_steps−tp,1)` | CBF 보정 호출 수 |
| `alpha` | `2.0` | 보정 명목 벡터장 크기 |
| `rho` | `0.9` | 유한 시간 CBF 지수 |
| `delta` | `0.01` | 정규화 장벽 여유 |
| `eps` | 자동 계산 | 보정 강도 |

기본 자동 `eps`는 `0<rho<1`, `delta>0`에서 다음 식을 쓴다.

\[
b_* = \frac{\delta}{1-\rho},\qquad
\varepsilon=\frac{0.9}{\Delta\tau}
\frac{b_*}{(b_*-\delta)^\rho}.
\]

이는 완화되지 않은 선형 장벽의 feasible 영역에서 Euler 한 스텝이 경계를 넘지 않게 하려는 설정이다. 기본 `ode_steps=10`이면 예측 1회와 보정 9회, 총 신경망 10회 및 CBF-QP 9회이며 `eps≈7.07`이다. `tp`, `tc`를 직접 지정하면 총 호출 수는 달라질 수 있다.

```bash
python fm_pc/eval_pushing.py \
  --run-dir fm_baseline/results/pushing_baseline \
  --ckpt baseline_ema.pt --sampler pc \
  --constraints action_box,tracking \
  --pc-params '{"tp": 1, "alpha": 2.0}' \
  --execute-steps 8
```

`--sampler` 기본값은 `euler`라서 `--constraints`만 지정하면 PC가 켜지지 않는다. PC 제약 목록은 기본적으로 `--constraints`를 따르며 `--pc-constraints`로 별도 지정할 수 있다. 수치는 `--constraint-params '{"a_max": 0.008, "track_d": 0.08}'`처럼 덮어쓴다.

생성한 전체 계획 중 앞 `execute_steps`개 행동을 실행하고 다시 관측해 재계획한다. 실행은 `des ← des+a`이며 **각 행동 실행 직전에 별도 CBF를 다시 계산하지 않는다.** Pushing CLI의 `execute_steps` 기본값은 1, Inserting은 8이다. 이 문서의 예시는 8을 명시했다.

중간 노이즈 샘플 자체를 투영하는 방식과의 비교는 [Projection 문서](Projection_d3il.md)를 참고한다.
