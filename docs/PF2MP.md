# PF2MP: 논문 정리와 D3IL 적용안

근거는 논문 *Towards Safe Imitation Learning via Potential Field-Guided Flow Matching*(PDF는 저장소에 넣지 않고 로컬 `docs/`에 둔다)의 arXiv:2508.08707v1, 2025-08-12 버전이다. 방법은 III–IV절(pp. 2–4), 실험은 V절(pp. 5–7)을 참조한다. 아래에서 **논문의 방법**, **Gaussian KDE를 선택했을 때의 수식 전개**, **이 저장소에 적용하기 위한 제안**을 구분한다. 현재 구현은 `fm_constraint/pf2mp.py`와 `policy.py`에 있으며, D3IL 적용은 아래 5~6절과 `PLAN.md` 8절을 따른다. 논문 결과와 이 저장소의 기능 검증 결과는 구분한다.

## 1. 핵심 이해

사용자가 이해한 “시연 데이터에 Gaussian kernel을 놓아 확률 밀도 함수(PDF)를 추정하고, Flow Matching 샘플링마다 potential field gradient를 적용한다”는 설명이 핵심적으로 맞다. 정확히는 다음 과정이다.

1. 성공 시연으로 관측 조건부 FM 정책을 학습한다.
2. 같은 시연의 점들로 KDE 밀도장을 만들고, 밀도와 시연 영역까지의 거리로 potential을 구성한다.
3. 추론 시 **매 Euler 적분 단계에서** FM 벡터장에 potential의 gradient를 더한다. 하나의 행동 시퀀스를 완성한 뒤 한 번 보정하는 방식과 구분한다.

논문은 KDE의 일반적인 kernel 식을 제시한다. Gaussian은 이를 구현하는 자연스러운 선택이지만, 본문에 kernel 종류·bandwidth 등 재현에 필요한 설정이 모두 명시된 것은 아니다. 또한 gradient 대상은 PDF 자체가 아니라 **log-density와 거리 penalty로 구성한 potential**이다.

직관은 시연이 많은 영역으로 생성 궤적을 유도하는 것이다. 시연이 적은 영역을 위험의 대리 지표로 사용하며, 밀도 자체가 장애물 지도나 충돌 확률은 아니다.

## 2. 논문의 기본 FM 정책

생성 시간을 \(t\in[0,1]\), 실제 제어 시점을 \(\tau\), 예측 길이를 \(T_a\)로 둔다. 논문은 현재 관측 \(o^\tau\)를 조건으로 행동 시퀀스를 생성한다.

\[
A_1=(a^\tau,\ldots,a^{\tau+T_a-1}),\qquad
A_0\sim\mathcal N(0,I),\qquad
A_t=(1-t)A_0+tA_1.
\]

학습 목표는 다음과 같다(논문 식 (7)–(8)).

\[
\mathcal L_{\mathrm{FMP}}
=\mathbb E\left[\left\|v_\theta(A_t,t;o^\tau)-(A_1-A_0)\right\|_2^2\right].
\]

기본 Euler 샘플러는 \(\Delta t=1/N\)에 대해

\[
A_{t+\Delta t}=A_t+\Delta t\,v_\theta(A_t,t;o^\tau)
\]

로 적분한다. Potential은 이 학습 손실에 들어가지 않는다. 기존 FM 가중치를 고정하고 KDE를 별도로 준비하여 추론 보정을 붙일 수 있다.

## 3. KDE와 potential field

### 3.1. Gaussian KDE를 선택하는 경우

KDE에 넣을 한 점을 \(z\in\mathbb R^d\), 시연 점들을 \(z_1,\ldots,z_M\)이라 하자. 논문에서는 이 변수를 \(a\)로 쓴다. 여기서는 D3IL 행동 증분과 구분하기 위해 \(z\)로 표기한다.

등방성 Gaussian kernel과 bandwidth \(b>0\)를 선택하면

\[
\widehat p(z)=\frac{1}{M(2\pi)^{d/2}b^d}
\sum_{i=1}^M\exp\left(-\frac{\|z-z_i\|_2^2}{2b^2}\right).
\]

각 데이터 점을 중심으로 Gaussian을 놓은 뒤 평균낸 밀도다. 하나의 Gaussian으로 데이터 전체를 fitting하는 것은 아니다. 위 식은 다차원 정규화까지 포함한 전개이며, 논문 식 (6)은 \(1/(Mh)\) 형태의 일반 KDE 식을 제시한다.

- 작은 \(b\): 시연 경로를 세밀하게 따르지만 밀도장이 좁고 gradient가 급해질 수 있다.
- 큰 \(b\): 부드럽지만 장벽을 사이에 둔 시연 영역들이 연결될 수 있다.
- 데이터의 단위·채널별 scale에 따라 거리가 달라지므로 좌표계와 정규화를 함께 정해야 한다.

Gaussian KDE의 log-density gradient는 다음처럼 계산된다.

\[
w_i(z)=\frac{\exp(-\|z-z_i\|^2/(2b^2))}
{\sum_j\exp(-\|z-z_j\|^2/(2b^2))},\qquad
\nabla_z\log\widehat p(z)
=\frac{\sum_iw_i(z)z_i-z}{b^2}.
\]

즉, 가까운 시연 점들의 가중 평균 쪽으로 향한다. \(\nabla\widehat p\)와는 밀도값만큼 크기가 다르다. 구현에서는 Gaussian 값을 직접 더하고 log를 취하기보다 `logsumexp`와 softmax를 사용하면 저밀도 영역의 underflow를 줄일 수 있다.

### 3.2. 논문의 potential과 부호

논문 식 (10)은 다음과 같다.

\[
\phi(z)=\log\widehat p(z)+\alpha\,d(z,\mathcal H),
\qquad \alpha<0,
\qquad \mathcal H=\{z:\widehat p(z)>\eta\}.
\]

\(\eta\)는 밀도 threshold이고, \(\mathcal H\)는 시연이 충분히 덮은 영역이다. \(d(z,\mathcal H)\)는 그 영역까지의 거리 penalty다. 따라서

\[
\nabla\phi(z)=\nabla\log\widehat p(z)+\alpha\nabla d(z,\mathcal H).
\]

양의 guidance weight를 사용할 때 **\(+\nabla\phi\)** 를 더한다. Log-density를 증가시키고, 음의 계수 \(\alpha\)가 붙은 거리는 감소시키는 방향이다. Potential을 비용으로 정의하여 \(-\nabla U\)를 쓰는 관례와 부호를 혼동하지 않아야 한다.

본문은 \(d\)의 구체적 계산법, \(\eta\), \(\alpha\), bandwidth, 전체 태스크의 KDE 구현 세부를 충분히 제공하지 않는다. 따라서 거리장을 grid에서 계산할지, level set의 최근접점을 찾을지 등은 추가 구현 선택이다. 예를 들어 Euclidean 거리 \(d(z,\mathcal H)=\inf_{y\in\mathcal H}\|z-y\|\)를 선택하면 영역 내부에서 0이며, 경계나 최근접점이 여러 개인 곳에서는 미분 가능성을 처리해야 한다. 이 선택을 논문에 명시된 구현으로 간주해서는 안 된다.

### 3.3. 매 FM 적분 단계에 적용

논문은 각 시점의 gradient를 행동 시퀀스에 쌓는다(IV-B절; 원문에서 식 번호 (11)이 중복된다).

\[
\Phi(A)=\left(\nabla\phi(a^\tau),\ldots,
\nabla\phi(a^{\tau+T_a-1})\right).
\]

최종 업데이트는 식 (12)와 Algorithm 2의

\[
\boxed{A_{t+\Delta t}=A_t+\Delta t\left[v_\theta(A_t,t;o)+\lambda\Phi(A_t)\right]}
\]

이다. Guidance는 **갱신 전의 동일한 \(A_t\)** 에서 평가하고, FM 벡터장과 함께 \(\Delta t\)를 곱한다. \(\lambda=0\)이면 기본 FMP로 돌아간다. 원문은 고정 \(\lambda\)로 매 단계 적용하며, 후반 단계에만 적용하거나 시간별 weight를 쓰는 것은 별도 변형이다.

## 4. 논문 실험에서 확인한 범위

| 태스크 | 기본 FMP 충돌률 | PF2MP 충돌률 | 비고 |
|---|---:|---:|---|
| Maze Medium | 0.33 ± 0.12% | 0.07 ± 0.07% | 성공률 99.7 → 99.6% |
| Maze Large | 9.2 ± 0.8% | 3.2 ± 0.5% | 성공률 96.6 → 96.4% |
| Fetch Reach | 34.0 ± 5.6% | 18.0 ± 4.4% | 성공률 85.0 → 84.0% |
| Fetch Push | 30.0 ± 3.6% | 25.0 ± 2.6% | 성공률 50.0 → 51.7% |
| Genesis Handwriting | 25.33 ± 2.52% | 0.67 ± 0.58% | joint-space 제어 |
| Unitree Z1 Handwriting | 4/10회 | 0/10회 | 소규모 실제 로봇 평가 |

근거는 Table I–II다. 충돌률은 **장애물과 한 번이라도 충돌한 trajectory의 비율**이며, 시점별 접촉 비율과 다르다. Maze·Fetch의 성공률은 실행 중 충돌 여부와 독립적으로 측정한다.

Maze에서는 계산량을 줄이기 위해 시연 점 3,000개를 무작위 추출했고, \(N=5\), \(\lambda=0.8\)을 사용했다. Genesis는 \(N=5\), \(\lambda=0.5\), 실제 Z1은 \(\lambda=0.5\)다. 이 값들을 정규화와 행동 의미가 다른 D3IL에 그대로 가져오는 근거는 없다. Fig. 3에서는 큰 \(\lambda\)가 오히려 overshoot와 충돌 증가·성공률 감소를 유발했다.

## 5. D3IL에 붙일 때 가장 중요한 차이

현재 저장소의 FM은 논문의 행동 시퀀스 전용 표기와 달리 **상태·행동을 함께 생성**한다. 기준 코드는 [baseline Policy](../fm_baseline/eval_pushing.py), [오프라인 샘플러](../fm_baseline/sample.py), [통합 정책](../fm_constraint/policy.py)이다.

| 항목 | 현재 저장소 |
|---|---|
| 생성 변수 | \(X_h=[a_h,s_h]\), Pushing 14차원 / Inserting 18차원 |
| 행동 | \(a_h=[\Delta q_x,\Delta q_y]\), 목표 TCP 위치 증분(m) |
| 관측 | 목표 TCP XY, 실제 TCP XY, 각 블록 XY와 sin/cos yaw |
| 예측 길이 | 기본 \(H=16\) |
| 조건 | 첫 행의 관측 \(s_0\)를 현재 실제 관측으로 고정 |
| 정규화 | 학습 split 기반 채널별 center/scale |
| 통합 설정의 생성 | Predict Euler 10회, 앞 8개 행동 실행 후 재계획 |

### 5.1. 행동 증분의 KDE만으로는 공간 장벽을 표현하기 어렵다

\(\widehat p(a)\)를 2차원 증분에 fitting하면 “시연에서 흔한 이동 방향·크기”를 추정한다. 같은 \([\Delta x,\Delta y]\)가 빈 공간에서는 허용되고 장벽 앞에서는 충돌할 수 있으므로, 이것만으로 Inserting의 장벽 위치를 구분하기 어렵다. 정지 행동이 많으면 영점 근처로 끌리는 효과도 발생할 수 있다.

논문의 Maze는 위치 궤적을 예측하며 정확하게 추종하는 저수준 제어기를 가정한다. 따라서 논문의 \(a\)라는 표기를 D3IL의 `actions` 배열과 바로 동일시하면 안 된다.

### 5.2. 현재 구현: 미래 상태 위치에만 energy gradient 적용

태스크별 전체 train 시연에서 목표 TCP XY의 Gaussian KDE를 만든다. 초기 상황이 달라도 같은 태스크의 train 모집단을 합치며 val/test는 제외한다. 계산 근사로 최대 3,000점을 고정 seed로 추출한다.

PF 대상은 생성된 미래 상태의 목표 위치 `X[:,1:,2:4]`다. Action 증분 채널 `X[:,:,:2]`, 첫 관측, 실제 TCP·블록·방향 채널의 PF 가산항은 모두 0이다. 에너지는 이 미래 위치들의 potential 합에 음수를 붙여 정의한다.

$$
\Psi(S)=\sum_{h=1}^{H-1}\phi(q_h^{state}),\qquad E=-\Psi,
\qquad v_{guided}=v_\theta-\lambda\nabla E.
$$

Raw 목표 위치가 $q=c_q+S_q\widetilde q$이면 정규화 위치 gradient는 $S_q^\top\nabla_q\phi$다. **행동 누적으로 위치를 재구성하여 action으로 역전파하지 않는다.** 이전 문서의 행동 chain rule 제안은 사용자 지정에 따라 이 방식으로 교체했다.

같은 갱신 전 계획에서 FM(④ RTC가 켜져 있으면 RTC 보정 벡터장)과 PF를 평가하고 Euler한 뒤, 선택적으로 ① 행동 Projection을 수행한다. 이후 ⑤ LiPo와 ③ 실행 직전 필터가 이어진다([METHODS.md](METHODS.md)). PF가 바꾼 미래 상태는 다음 FM 평가에서 행동에 간접 영향을 줄 수 있다. 한 번만 Euler하면 PF에 의한 행동 변화는 없으며, 상태 에너지 개선이 실행 행동의 안전 개선을 보장하지 않는다. 명시 제약은 계속 행동 누적 목표에 적용하고 생성 상태와의 차이를 기록한다.

## 6. 캐시와 실행

`fm_constraint/pf2mp.py`에서 전체 train의 shape·유한값·시간·시연 내 연결·행동 대응·sin/cos·분할 정합성을 검사하고 Gaussian support와 거리장을 만든다. 밀도는 해석적 Gaussian 혼합으로 평가하고 거리항은 threshold를 넘는 2D grid 점 집합까지의 최근접 거리로 근사한다. 이는 논문에서 상세히 지정하지 않은 거리장에 대한 구현 선택이다.

PF on의 추론 시작 시 `data/<task>_sincos_v1/pf2mp_cache/<fingerprint>.npz`를 준비한다. Data/manifest hash, bandwidth, support 추출, grid·threshold 설정으로 캐시를 구별한다. 없으면 생성하고 있으면 검증 후 재사용하며, PF off에서는 읽거나 만들지 않는다. Weight·거리항 계수만 바꾸면 캐시를 재사용한다. 구현 이력과 검증 기록은 [PLAN](PLAN.md)과 `fm_constraint/results/pf2mp_validation.json`에 있다.

```bash
python -m fm_constraint.sample --config configs/pushing.yaml --pf2mp --projection \
  --set sample.n_windows=4
python -m fm_constraint.sample --config configs/inserting.yaml --pf2mp --lipo \
  --set methods.pf2mp.weight=0.01 --set sample.n_windows=4
```

설정은 schema v4(`methods.pf2mp`)다. Gaussian score와 거리항은 해석적으로 계산하므로 FM 모델에 대한 backpropagation이 필요 없다. 기본 수치는 초기 후보값으로 전체 태스크 성능을 검증한 최적값이 아니다.

## 7. 평가와 남은 설계상의 주의점

동일 checkpoint·normalizer·context·noise·Euler 횟수·실행 chunk로 기본 FM / log-density만 적용 / 완전한 potential을 비교한다. Bandwidth, threshold, 거리 계수, guidance weight, 거리 근사법, support 수·추출 방법을 기록하고 조정용 validation과 최종 평가를 분리한다.

성공률, episode별 금지 접촉률, 성공하면서 무충돌인 비율, 행동 변화, 명령 가속도, 모드 다양성, 생성 시간을 측정한다. 필요한 손·블록 접촉을 실패로 세지 않는다. 접촉 정보가 없으면 충돌률을 추정하지 않고 workspace·가속도 위반과 구분한다. 밀도·거리 평가 비용은 모델 호출 비용과 별도로 보고한다.

- 미방문 영역이 반드시 위험하지는 않다. 시작 분포·목표·체류 시간도 밀도에 반영되며 고밀도가 물리 안전의 증명은 아니다.
- 주변 XY KDE는 블록 배치·접촉 문맥을 구분하지 않는다. 상대 좌표·관측 조건부 밀도는 추가 설계다.
- Gaussian의 꼬리는 장애물 안으로도 뻗는다. 좁은 통로에서는 bandwidth에 따라 벽을 가로지르는 유도가 생길 수 있어 밀도장을 실제 장애물과 겹쳐 확인한다.
- 생성 초기의 위치는 데이터에서 멀 수 있다. $1/b^2$ 효과로 gradient가 커질 수 있으므로 유한값·gradient norm을 기록한다. Clipping·시간 schedule은 별도 변형으로 비교한다.
- 상태 위치 guidance는 다음 FM 평가를 거쳐 행동에 영향을 주므로 상태 에너지와 행동 누적 위치의 차이를 함께 평가한다. 실제 TCP·블록·전체 링크의 안전은 별도 검증 대상이다.

현재 검증 단위는 **train 목표 XY KDE → 미래 상태 위치 potential → action 가산항 0 → 매 Euler 단계 보정**이다.
