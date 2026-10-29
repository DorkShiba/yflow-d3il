# Y-Flow: Flow Matching 기반 로봇 정책에서 Smoothness × Safety 제약의 결합 연구

> AI Capstone Design · 서준혁, 임성민, 정재엽, 조형빈 · Yonsei University
> 근거 자료: `constraint_flow.pdf`(Method 초안, 7개 기법), `meeting_1008_handout.pdf`(2026.10.08 미팅, 5개 기법으로 정리 + 예비 결과)

---

## 1. 연구 목표

**Flow matching으로 학습한 로봇 조작 정책이 "부드럽고 자연스러우면서(smooth) 동시에 모든 제약을 만족하는(safe)" 행동만 생성하도록 만드는 것.**

이를 위해 구체적으로 다음을 목표로 한다.

1. **정책 smoothness 계열**(CAPS, L2C2)과 **safety 제약 계열**(Projection, Predict-then-Correct, Action Filter)을 하나의 flow matching 정책 위에 결합했을 때의 효과를 **최초로 체계적으로 측정**한다.
2. 결합이 **상보적(complementary)인지, 상충적(conflicting)인지**를 정량 지표 Δ로 판정한다.
3. 5개 기법의 전체 조합(2⁵ = 32가지) 중 **가장 성능이 좋은 조합을 찾아 이를 "Y-Flow"로 정의**하고, 이후 이를 최적화·확장한다.

---

## 2. 연구의 필요성 (Motivation)

### 2.1 배경: Diffusion → Flow Matching

| | Diffusion Policy | Flow Matching Policy |
|---|---|---|
| 생성 방식 | 노이즈를 조금씩, 아주 여러 번 제거 | 각 점에서 "이동할 방향" $v_\theta(x,t)$를 학습 |
| 샘플링 경로 | 많은 수의 작고 구불구불한 step | 적은 수의 거의 직선적인 step |
| 업데이트 | 반복적 denoising | $x_{k+1} = x_k + \Delta t\, v_\theta(x_k, t_k)$ |

Flow matching은 적은 step으로 빠르게 행동을 생성할 수 있어 실시간 로봇 제어에 유리하다. 그러나 **생성 모델 자체는 "가능한/안전한 제어"인지를 보장하지 않는다.**

### 2.2 문제: 현재 정책 출력의 두 가지 결함

학습된 정책을 그대로 실행하면(raw policy output):

- **Motion – jittery**: 행동이 떨리고 급격히 바뀜
- **Safety – collision**: 장애물을 인지하지 못하고 충돌

우리가 원하는 것은 **smooth & natural motion + every constraint satisfied** 이다.

### 2.3 기존 연구의 한계: 두 계열이 따로 연구됨

| 계열 | 대표 기법 | 단독 적용 시 결과 |
|---|---|---|
| Policy smoothness | CAPS, L2C2 | Smooth ✅ / Safe ❌ (부드럽지만 충돌) |
| Safety constraints | Projection, Predict-Correct, Filter | Smooth ❌ / Safe ✅ (안전하지만 떨림) |

- 각 계열은 **각자 따로** 연구되어 왔고, **두 계열 간 결합은 연구된 적이 없다.**
- 따라서 둘을 함께 쓰면 **서로 보완하는지, 서로를 상쇄하는지 알 수 없다.**

### 2.4 왜 중요한가

- 실제 로봇은 smoothness와 safety를 **동시에** 필요로 하므로, 결국 하나의 정책 위에 두 종류의 제약이 **쌓이게(stacked)** 된다.
- 만약 두 제약이 충돌한다면, **하나를 추가하는 것이 다른 하나의 효과를 조용히 무효화**할 수 있다.
- Δ를 알면 **어떤 조합이 실제 배포할 가치가 있는지** 판단할 수 있다.

> 예시 (Constraint Interaction): Method A는 joint limit은 만족하지만 obstacle zone을 침범, Method B는 반대. A+B를 결합하면 둘 다 만족 — 이런 결합 이득을 수치로 측정하는 것이 핵심.

---

## 3. 연구 내용

### 3.1 핵심 지표: Complementarity Δ

$$
\Delta = S_{A+B} - \max(S_A,\ S_B)
$$

- $S_A, S_B$: 각 기법 단독 적용 시 점수, $S_{A+B}$: 결합 시 점수
- **Δ > 0 → 상보적(complementary)**: 결합이 최선의 단일 기법보다 낫다
- **Δ < 0 → 상충적(conflicting)**: 결합이 오히려 손해
- Δ는 **각 metric별로** 계산한다 (성공률, 위반율, smoothness 등)

### 3.2 방법 (Methods)

최종 실험 대상은 **Safety 3종(①②③) + Smoothness 2종(④⑤)** 이다.
(초안에는 ⑥ Spectral Normalization, ⑦ Weight Decay도 포함되어 있었으나 10/08 미팅 자료에서는 5개로 정리됨 → 3.2.3 참고)

#### 3.2.1 Safety 제약 계열 (추론 시 적용)

**① Projection at Every Step**
- **아이디어**: 매 샘플링 step 후 샘플을 제약 집합 위로 "다시 끌어다 놓는다".
- **수식**:
  $$x_{k+1} = \Pi_{\mathcal C}\big(x_k + \Delta t\, v_\theta(x_k, t_k)\big), \qquad \Pi_{\mathcal C}(y) = \arg\min_{z\in\mathcal C}\|z-y\|^2$$
  - $v_\theta$: 학습된 velocity field, $\Pi_{\mathcal C}$: 제약 집합 내 최근접점으로의 projection, $\mathcal C$: feasible set (joint limit, 장애물 없는 영역 등)
- **기대 효과**: 모든 출력이 제약 만족 / 추론 단계 plug-in, 재학습 불필요
- **Trade-off**: 분포 이탈(distribution drift), local trap (경계를 따라 계속 밀려나는 현상)
- **선행 연구**: DPCC (L4DC 2025), SafeDiffuser (ICLR 2025). D3IL에서 constraint+goal 달성률 0.07 → 0.98. DPCC는 실행할 projected candidate 선택 방식에 따라 R(random)/T(temporal consistency)/C(projection cost) 변형이 있음.

**② Predict, then Correct**
- **아이디어**: 먼저 자유롭게 생성(draft)한 뒤, 최종 경로만 CBF(Control Barrier Function)로 보정한다. 중간 latent는 projection하지 않는다.
- **수식**:
  $$\text{Predict: } \hat x_1 = x_0 + v_\theta(x_0, 0)$$
  $$\text{Correct: } u^\star = \arg\min_u \|u - \alpha(1-t)\, v_\theta(x_t,t)\|^2 \ \ \text{s.t.}\ \ \nabla b(x_t)^\top u \ge -\gamma\,\mathrm{sign}(b)\,|b(x_t)|^\rho$$
  - $\hat x_1$: 한 번의 자유 flow step으로 얻은 draft 경로, $\alpha(1-t)$: 끝으로 갈수록 사라지는 vanishing field (경로가 안정적으로 수렴), $b(x)\ge 0$: CBF 조건을 통해 유한 시간 내 도달하는 safe set
- **기대 효과**: 중간 latent 보존 / local trap 없음, 약 60× 빠름
- **Trade-off**: draft 품질에 의존
- **선행 연구**: SafeFlowMatcher (ICLR 2026), HardFlow (ICLR 2026). Maze2D trap rate 0% (기존 69–100%). 7-DoF 팔 block stacking에서 joint-limit 제약 하 계획.

**③ Filter the Executed Action (Safety Filter)**
- **아이디어**: 정책은 그대로 두고, 실행 직전 각 행동을 **최소한으로만** 수정한다. 장애물에서 멀면 행동은 그대로 통과.
- **수식**:
  $$u^\star = \arg\min_u \|u - u_\pi\|^2 \quad \text{s.t.}\quad \nabla h(p)^\top \dot p(u) \ge -\alpha\, h(p)$$
  - $u_\pi$: flow matching 정책이 제안한 행동, $h(p)$: end-effector와 장애물 ellipsoid 간 거리, $\dot p(u)$: 행동이 만드는 end-effector 속도
- **기대 효과**: 재학습 불필요, 어떤 정책에도 plug-in / 실시간 (제어 step당 작은 QP 1회, 약 0.36 ms)
- **Trade-off**: 근시안적(myopic), 인식(perception) 오차에 취약
- **선행 연구**: AEGIS / VLSA (arXiv 2025), Safe Streaming Flow (arXiv 2026). SafeLIBERO CAR 18.7% → 77.9%. 실제 Franka + π0.5-DROID에서 안전성·성공률 동시 개선.

#### 3.2.2 Policy Smoothness 계열 (학습 시 적용)

**④ CAPS: Conditioning for Action Policy Smoothness**
- **아이디어**: 학습 시 regularizer로, 시간에 따른 행동 변화와 상태 노이즈에 따른 행동 변화를 벌점화한다.
- **수식**:
  $$L_T = \mathbb E\,\|\pi(s_t) - \pi(s_{t+1})\|_2, \qquad L_S = \mathbb E\,\|\pi(s) - \pi(\bar s)\|_2$$
  $$\mathcal L = \mathcal L_{\text{FM}} + \lambda_T L_T + \lambda_S L_S, \qquad \bar s \sim \mathcal N(s, \sigma_S^2 I)$$
  - $L_T$: 시간적 항(연속 상태 간 행동 변화), $L_S$: 공간적 항(작은 상태 섭동에 대한 행동 변화), $\bar s$: 현재 상태 주변의 Gaussian 이웃
- **기대 효과**: 좌우 명령 뒤집힘(flip) 감소 / action change와 jerk를 직접 감소 (toy rollout에서 총 action change 약 13× 감소)
- **Trade-off**: 급격한 회피 동작이 무뎌질 수 있음
- **선행 연구**: Mysore et al. (ICRA 2021). 실제 쿼드로터에서 전력 80% 이상 절감, 2024 smoothness benchmark에 포함.

**⑤ L2C2: Local Lipschitz Continuity Constraint**
- **아이디어**: 행동 민감도(Lipschitz 상수)를 **실제로 방문한 상태 근처에서만** 작게 유지한다. 데이터가 없는 영역에는 제약을 걸지 않으므로, 과제상 필요한 **급격한 전환(sharp switch)은 보존**된다. (전역 smoothness는 이 전환을 흐려버림)
- **수식**:
  $$\text{L2C2: } C_\pi = \mathbb E_{\tilde s\in U_s}\!\left[\frac{d_\pi(\pi(\cdot\mid s),\ \pi(\cdot\mid \tilde s))}{d_U(s,\tilde s)}\right]$$
  $$\text{FM adaptation: } R_{\text{local}} = \mathbb E\!\left[\frac{\tfrac12\|g_\theta(s,z) - g_\theta(\tilde s,z)\|_2^2}{d_U(s,\tilde s;\ s')}\right]$$
  - $\tilde s$: 실제 전이 $s\to s'$로부터 만든 이웃 상태, $d_U$: 입력 거리(멀수록 약한 벌점), $g_\theta(s,z)$: **공유된 latent $z$** 로 flow matching이 생성한 행동
  - → 확률적 생성 정책에 맞게, 같은 noise $z$를 고정해 두 상태의 출력 차이를 비교하는 것이 우리 쪽 FM 적응 포인트
- **기대 효과**: 관측 노이즈에 덜 민감(action spread 감소) / 국소적이므로 표현력 유지
- **Trade-off**: 장애물 반응이 느려질 수 있음
- **선행 연구**: Kobayashi (IROS 2022), PyBullet (Reacher 포함). LipsNet과 결합 시 2024 benchmark 최고 성능. 실제 로봇 사례: LipsNet++ (ICML 2025).

#### 3.2.3 초안에서 검토했으나 최종 실험에서 제외된 기법

| 기법 | 수준 | 핵심 | 장점 | Trade-off | 출처 |
|---|---|---|---|---|---|
| ⑥ Spectral Normalization | 아키텍처 | $\widehat W = c\,W/\sigma_{\max}(W)$, $\mathrm{Lip}(f)\le\prod_\ell\|W_\ell\|_2$ — 각 layer의 입력 증폭을 c로 제한 | 추가 loss 없음, 연산 부담 거의 없음 | 너무 강하면 성공률 하락 | Miyato et al. (ICLR 2018), Humanoid LCP (arXiv 2410.11825) |
| ⑦ Weight Decay (AdamW) | 옵티마이저 | $\theta_{k+1} = (1-\eta\lambda_{\text{WD}})\theta_k - \eta\,\hat m_k/(\sqrt{\hat v_k}+\epsilon)$ | 단순한 함수, 적은 demo에서 일반화 개선 | 너무 강하면 underfit | Loshchilov & Hutter (ICLR 2019) |

> 참고: 10/08 자료에서 이 둘이 빠진 명시적 이유는 적혀 있지 않다. 실험 수(2⁷=128 → 2⁵=32)를 줄이고 "smoothness vs safety" 두 계열 비교에 집중하려는 의도로 보이며, 팀 내 확인이 필요하다.

---

## 4. 실험 설계

### 4.1 데이터셋 / 환경

두 후보 모두 state-action demo와 시뮬레이터를 제공하므로 **rollout으로 제약 만족 여부를 직접 검사**할 수 있다.

| | D3IL (Jia et al., ICLR 2024) | robomimic (Mandlekar et al., CoRL 2021) |
|---|---|---|
| Medium | Avoiding, Pushing, Aligning | Can |
| Hard | Sorting, Stacking, Inserting | Square, Transport |
| Very hard | Arranging | Tool Hang |
| Easy | — | Lift |
| 특징 | 7개 과제, 과제당 다양한 정답 해(multi-modal) | 5개 sim 과제, robosuite replay 연동 |
| 추가할 제약 | keep-out zone, obstacle margin, step size | EE region, carry pose, action range |

(난이도는 공식 구분이 아닌 팀의 정성적 분류)

**현재 진행**: **D3IL Pushing** — 각 블록(빨강/초록)을 같은 색 구역으로 밀어 넣기
**다음 단계**: **D3IL Inserting** — barrier 주변의 블록 3개 삽입

### 4.2 평가 지표 (Metrics)

| 구분 | 지표 | 정의 |
|---|---|---|
| Task | Success rate | 두 블록 모두 목표 구역 안에서 종료 (D3IL 기준) |
| Safety | Constraint violation rate | 제약(keep-out zone, step size, action range)을 위반한 step의 비율 |
| Smoothness | Action change / jerk | rollout 동안 평균 $\|a_t - a_{t-1}\|$ 및 jerk (낮을수록 smooth) |
| Diversity | Behavior entropy | D3IL score ∈ [0,1]: 제약이 4가지 push 순서(multi-modality)를 붕괴시키는가? |
| Combination | Complementarity Δ | $\Delta = S(A{+}B) - \max(S(A), S(B))$, metric별 계산 |

### 4.3 Full Ablation 계획

- 기법 ①–⑤의 모든 on/off 조합: **2⁵ = 32 runs** (baseline 포함)
- 자원: RTX 3090 × 2, 4인, 약 2주

| 구성 | baseline | 1개 | 2개 | 3개 | 4개 | 전체 5개 |
|---|---|---|---|---|---|---|
| run 수 | 1 | 5 | 10 | 10 | 5 | 1 |
| 상태 | ✅ 완료 | ✅ 완료 | ⏳ | ⏳ | ⏳ | ⏳ |

**진행 현황: 6 / 32 완료** (baseline + 각 제약 단독). 남은 26개 조합이 다음 작업.

### 4.4 예비 결과 (Demo, baseline vs. 단독 safety 기법)

동일 시작 조건에서의 단일 episode 시각화. 표의 값은 "Goal 3D" 거리(목표까지 남은 3D 오차, 낮을수록 좋음)이며 통계적 결과가 아닌 정성적 관찰용이다.

| 기법 | c004_t3 Goal 3D | c021_t0 Goal 3D |
|---|---|---|
| Baseline | 95.1 mm | 58.1 mm |
| ① Projection | 32.1 mm | 29.2 mm |
| ② Predict-then-Correct | 35.2 mm | 28.4 mm |
| ③ VLSA (Filter) | 95.1 mm | 35.6 mm |

- ①, ②는 두 시작 조건 모두에서 baseline 대비 목표 오차가 크게 줄었다.
- ③은 시작 조건에 따라 효과가 달랐다 (c004_t3에서는 baseline과 동일 → 해당 episode에서는 필터가 개입하지 않았을 가능성).
- 궤적 자체는 여전히 떨림(jitter)이 남아 있어, smoothness 계열과의 결합 필요성을 뒷받침한다.

---

## 5. 기대 결과 및 해석 시나리오

| 시나리오 | 의미 | 해석 |
|---|---|---|
| 제약 1–2개 조합이 최고 | 제약들이 충돌하여 상쇄 (Constraint 1 ↔ Constraint 2) | 제약을 많이 거는 것보다 **움직임의 자유도가 더 중요** |
| 제약 3개 이상 조합이 최고 | 제약들이 하나의 flow로 합쳐짐 → Combined motion | 찾은 최적 조합을 결과로 삼고 **추가 최적화** |

- 32개 조합 전체를 성능 순으로 순위화하고, **가장 성능이 좋은 조합 = Y-Flow**로 정의한다.
- 부수적으로, 각 pair에 대한 Δ 표를 통해 **"어떤 safety 기법이 어떤 smoothness 기법과 궁합이 좋은가"** 라는 실용적 가이드를 제공할 수 있다.

---

## 6. 이후 연구 목표 (Next: Optimise Y-Flow)

1. **남은 26개 조합 실험 완료** → 32개 전체에 대해 metric별 점수와 Δ 산출, 순위화
2. **Y-Flow 확정**: 최고 성능 조합 선정
3. **가중치 튜닝 (Tune the weights)**: safety와 smoothness 강도($\lambda_T, \lambda_S$, CBF의 $\alpha, \gamma$ 등)를 sweep으로 최적화
4. **샘플링 step 축소 (Fewer sampling steps)**: 제약 만족을 유지하면서 더 빠르게 실행 (flow matching의 장점 극대화)
5. **더 어려운 과제로 확장 (Harder task)**: Y-Flow를 **D3IL Inserting**으로 이전, 필요 시 robomimic(Lift → Can → Square)으로 일반화 검증

### 검토가 필요한 열린 질문

- ⑥ SN, ⑦ WD 제외 여부 확정 및 그 근거 정리
- ①/② 처럼 "샘플링 과정"에 개입하는 기법과 ③ 처럼 "실행 행동"을 수정하는 기법을 동시에 쓸 때 적용 순서
- 단일 seed 결과의 분산 문제 → 조합별 seed 수 및 평가 episode 수 결정
- Behavior entropy로 측정한 multi-modality 붕괴가 safety 제약(특히 ① Projection)에서 크게 나타나는지

---

## 참고 문헌 (자료에 인용된 것)

- Römer, von Rohr, Schoellig, "Diffusion Predictive Control with Constraints (DPCC)," L4DC 2025
- SafeDiffuser, ICLR 2025
- Yang, Jang, Han, "SafeFlowMatcher: Safe and Fast Planning using Flow Matching with CBFs," ICLR 2026
- HardFlow, ICLR 2026
- Hu, Liu et al., "VLSA: VLA Models with Plug-and-Play Safety Constraint Layer (AEGIS)," arXiv 2512.11891
- Safe Streaming Flow, arXiv 2026
- Mysore et al., "Regularizing Action Policies for Smooth Control with Reinforcement Learning (CAPS)," ICRA 2021
- Kobayashi, "L2C2: Locally Lipschitz Continuous Constraint," IROS 2022
- Song et al., "LipsNet++: Unifying Filter and Controller into a Policy Network," ICML 2025
- Miyato et al., "Spectral Normalization for GANs," ICLR 2018
- Chen et al., "Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies," arXiv 2410.11825
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization," ICLR 2019
- Jia et al., "D3IL," ICLR 2024 / Mandlekar et al., "robomimic," CoRL 2021
