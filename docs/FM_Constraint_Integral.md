# FM Constraint 통합 방침

이 문서는 `fm_projection`, `fm_pc`, `fm_constraint`를 통합하기 위한 **설계 명세**다. 아래의 통합 API·설정·새 제약은 구현 예정이며, 현재 코드에서 이미 지원하는 기능과 구분한다. 기존 FM 모델을 고정한 training-free 추론 방법의 통합을 대상으로 한다. 수식은 $$으로 감싸 작성한다. inline은 $으로 감싼다.

## 1. 통합의 목표와 확정할 원칙

세 방법을 하나의 정책·평가 경로에서 독립적으로 켜고 끈다. 제약의 정의는 공유하고, **어디에서 무엇을 보정하는지**를 방법별로 나눈다.

| 방법 | 통합 후 역할 | 신경망 추가 호출 |
|---|---|---|
| Projection | 생성 Euler 스텝 직후의 중간 계획 투영 | 없음 |
| VLSA | 생성 완료 후, PC 이전에 전체 행동 계획을 모델 없이 보정 | 없음 |
| Predict-then-Correct(PC) | 앞 단계의 계획을 받아 모델 벡터장과 CBF-QP로 반복 보정 | 보정 횟수만큼 |

통합 기본 순서는 초안대로 다음과 같이 고정한다.

```text
noise + 현재 관측 조건
  → Predict: FM Euler 생성 [Projection on이면 각 갱신 직후 투영]
  → clean plan
  → VLSA: 전체 행동 계획의 최소 수정 QP [선택]
  → corrected clean 1
  → PC Correct: 모델 재평가 + CBF-QP [선택]
  → corrected clean 2
  → 결과 검사·앞 K개 행동 실행·최신 관측으로 재계획
```

`clean`은 생성 시간이 끝난 계획이라는 뜻이다. 물리 제약 만족이나 실제 미래 상태의 정확성을 뜻하지 않는다. 모든 단계는 길이 `H`의 전체 계획을 다루고, 실제 실행은 앞 `K≤H`개 행동에 대해서만 한다.

**핵심 방침은 다음과 같다.**

1. 제약 기하와 파라미터는 한 곳에 정의한다. 세 방법이 서로 다른 workspace나 가속도 한계를 몰래 사용하지 않는다.
2. 통합에서 구현할 제약은 사각형 Possible Boundary, Acceleration Norm, 볼록 n각형 Object Envelope Tracking의 세 가지다. 두 태스크는 같은 수식을 사용하고 파라미터만 따로 갖는다.
3. 계획 보정은 raw 물리 단위에서 행동 채널만 수정한다. 현재 관측 조건은 유지한다.
4. 작업 영역과 Object Envelope는 직접 선형 부등식으로 정의한다. 가속도 L2 norm은 보수적인 다각형 근사로 선형화하여 공통 QP를 사용하고 실제 L2 값도 검사한다.
5. 현재 VLSA의 실행 시점 보정과 새 VLSA의 계획 보정을 명확히 구분한다. 실행 시점 재검사는 별도 옵션이다.
6. 세 방법을 모두 끄면 원래 baseline 경로를 사용한다. 비활성 방법의 투영·CBF·숨은 clipping을 실행하지 않는다.
7. 기존 코드와 결과는 재현 기준으로 보존한다. 통합 과정에서 바뀌는 수학적 의미는 새 버전으로 기록한다.

## 2. 현재 구현과 통합 시 바뀌는 점

| 항목 | 현재 구현 | 통합 방침 |
|---|---|---|
| Projection | `every` 또는 `final`, 행동·누적 목표에 투영 | 기본은 Predict 내부 `every`; `final`은 별도 비교 모드 |
| PC | 자체 Predict와 Correct를 모두 수행; 기본 1+9회 모델 호출 | Predict를 공통 생성부로 분리하고 Correct만 뒤에 연결 |
| VLSA | 각 행동 실행 직전에 최신 관측으로 2차원 행동 보정 | 기본은 PC 이전의 `2H`차원 계획 보정으로 확장 |
| 위치 제약 | Projection/PC는 학습 범위, VLSA는 YAML 사각형 | 태스크별 명시적 공통 workspace |
| 가속도 | 현재는 주로 한 행동의 이동량을 제한 | 행동 차분으로 정의한 목표 명령 가속도 norm을 세 방법에 공통 적용 |
| Tracking | 계획의 목표 위치를 제한 | 모든 블록을 포함하는 볼록 n각형 영역; n은 config로 지정 |
| 불가능한 경계 | PC/Projection은 일부 경계를 clip·중간값으로 변경 | 원래 경계를 묵시적으로 변경하지 않고 실패 상태 반환 |
| 평가 | 방법별 복제 루프와 공통 VLSA 루프가 혼재 | 하나의 평가 루프와 태스크 adapter 사용 |

현재 `fm_constraint`의 `geometry: circle`은 원형 장애물 바깥을 유지하는 수치 CBF 모드다. 통합의 사각형 작업 영역·볼록 다각형 Tracking과 별개인 legacy 기능으로 남긴다.

## 3. 공통 변수와 제약의 적용 대상

실제 제어 시점은 `t`, 생성·보정 시간은 `τ`, 계획 내부 인덱스는 `h=0,…,H−1`로 구분한다.

| 기호 | 정의 |
|---|---|
| `q₀` | 계획을 시작할 때 현재 목표 TCP XY 명령 |
| `p₀` | 같은 시점에 측정한 실제 TCP XY |
| `b_i` | 같은 시점에 측정한 i번째 블록의 XY 중심 |
| `a_h∈R²` | 목표 TCP XY 증분, 단위 m/제어 스텝 |
| `a_prev` | 계획 직전에 실제 실행한 목표 증분; episode 시작에서는 0 |
| `Δt` | 실제 제어 간격, 기본 `0.035 s`; 생성 시간 보폭과 구분 |
| `α_h` | `(a_h−a_(h−1))/Δt²`인 목표 명령의 이산 가속도(m/s²) |
| `A∈R^(2H)` | `[a₀,…,a_(H−1)]`를 쌓은 행동 계획 |
| `S_h` | `S_h A=a_h`가 되는 선택 행렬 |
| `C_h` | `C_h A=Σ_(j=0)^h a_j`가 되는 누적 행렬 |
| `q_h⁺=q₀+C_h A` | h번째 행동 실행 후 목표 TCP XY |
| `ξ` | 현재 관측·기하·이전 실행 행동을 묶은 제약 context |

모델은 정규화된 `[행동, 관측]` 전체의 벡터장을 출력하지만, 제약 계산은 raw 행동 `A`와 `q_h⁺`를 사용한다. 미래 관측 채널에 예측된 손 위치를 실제 손의 미래 위치로 취급하지 않는다.

현재 제어 인터페이스에서 직접 제한할 수 있는 값은 **목표 TCP 명령**이다. 실제 손 위치·팔 전체 링크의 도달 가능성은 동일한 것으로 볼 수 없다. 실제 손은 별도 지표로 측정한다. 실제 손 궤적의 강제 제약까지 확장하려면 검증된 추종·동역학 모델이 추가로 필요하다.

## 4. 통합 제약의 수학적 정의

모든 원래 제약은 `g_j(A;ξ)≤0`으로 표기한다. 경계를 포함하는 닫힌 집합을 사용하므로 초안의 `<`는 `≤`로 정한다. 실제 계산에는 수치 허용 오차를 적용한다.

### 4.1. Possible Boundary: 사각형 허용 작업 영역

작업 영역은 **축에 정렬된 사각형으로 고정**한다. 원형 workspace 옵션은 통합 스키마에서 제외한다. 설정 경계를 `ℓ,u`, 안쪽 margin을 `m_ws≥0`이라 두면

$$
\ell'=\ell+m_{ws}\mathbf1,\qquad u'=u-m_{ws}\mathbf1,
$$
$$
\ell'_i-q_{h,i}^{+}\le0,\qquad q_{h,i}^{+}-u'_i\le0.
$$

`q_h⁺=q₀+C_h A`이므로 계획 시점마다 네 개의 선형 부등식이다. 별도 기하 근사 없이 QP에 넣을 수 있다. 초기 경계는 기존 VLSA의 사각형을 재사용한다.

| 태스크 | 설정한 원래 XY 경계(m) | margin | 적용 경계(m) |
|---|---|---|---|
| Pushing | `[0.20,0.85] × [−0.40,0.50]` | `0.01` | `[0.21,0.84] × [−0.39,0.49]` |
| Inserting | `[0.00,0.85] × [−0.40,0.75]` | `0.01` | `[0.01,0.84] × [−0.39,0.74]` |

이는 **초기 연구용 명령 영역**이다. “로봇 팔이 물리적으로 도달 가능한 영역”으로 확정하려면 고정 z·자세에서 IK/관절 한계·환경 형상을 검사하여 내부 영역을 정해야 한다. 학습 데이터의 최솟값·최댓값만으로 도달 가능성을 선언하지 않는다. 현재 값은 `source: research_bounds`로 기록하고, 추후 검증된 경계로 교체한다.

### 4.2. Acceleration Norm: 목표 명령의 가속도

기존 Action Norm을 **연속 행동 사이의 변화로 정의한 가속도 norm**으로 교체한다. 행동 `a_h`는 위치 증분이므로, 고정 제어 간격 `Δt`에서

$$
v_h^{cmd}=\frac{a_h}{\Delta t},\qquad
\alpha_h^{cmd}=\frac{v_h^{cmd}-v_{h-1}^{cmd}}{\Delta t}
=\frac{a_h-a_{h-1}}{\Delta t^2}.
$$

요구할 원래 제약은

$$
\|\alpha_h^{cmd}\|_2\le\alpha_{max}
\quad\Longleftrightarrow\quad
\|a_h-a_{h-1}\|_2\le\alpha_{max}\Delta t^2.
$$

`α_max`의 단위는 **m/s²**다. `(a_h−a_(h−1))/Δt`는 속도 변화량에 해당하므로 가속도로 쓰지 않는다. `Δt=0.035 s`는 제어 간격이며 PC의 생성 시간 보폭 `Δτ`와 별개다.

- `h=0`의 `a_(−1)`에는 직전에 **실제 실행한** 행동 `a_prev`를 넣는다. 보정 전 명목 행동이나 이전 계획의 미실행 마지막 행동을 사용하지 않는다.
- 새 계획마다 `a_prev`를 0으로 초기화하지 않는다. 0은 episode 시작에서만 사용한다.
- 하나의 Predict→VLSA→PC 안에서는 `a_prev`를 고정한다. 실제 행동을 실행한 뒤 갱신한다.
- 오프라인 검사는 시연의 직전 행동을 함께 읽는다. 시연 첫 상태에서는 0을 사용하고 다른 시연의 마지막 행동을 연결하지 않는다.

L2 가속도 제한도 원판 제약이므로 그대로는 일반 선형 제약 QP가 아니다. **5.1절의 내접 다각형으로 보수적으로 근사**하여 세 방법 모두 같은 QP 면을 사용한다. 이는 L2 norm의 정확한 등가 변환은 아니다.

기존 이동량 `0.012 m`를 그대로 가속도 설정값으로 옮기지 않는다. `constraints.acceleration.max_m_s2`는 태스크별로 명시한다. 초기값 선택 시 학습 시연 안에서 `‖a_t−a_(t−1)‖/Δt²`의 분포와 episode 시작 조건을 계산하고, 선택한 한계에서 다각형 제약까지 만족하는 시연 비율을 확인한다. 값 확정 전 설정 예시의 `null`은 실행 가능한 한계가 아니다.

가속도 제한은 큰 행동이 계속되는 것을 직접 막지 않는다. **기존 한 스텝 이동량 제한은 기본 제약에서 제거**하며, 필요할 때만 별도 제약으로 검토한다. 이 가속도는 목표 명령의 이산 가속도다. 실제 로봇 팔의 가속도는 추종 동역학에 따라 달라진다.

### 4.3. Object Envelope Tracking: 모든 물체를 포함하는 볼록 n각형

허용 영역 자체를 **모든 블록 중심을 포함하는 정n각형**으로 정의한다. 정n각형은 볼록하므로 모든 변의 안쪽 반공간을 교집합으로 표현할 수 있다. 임의의 비볼록 n각형은 이 방식의 단일 볼록 QP에 사용할 수 없다.

`n=constraints.object_envelope.n_sides`는 정수 `n≥3`으로 설정한다. 법선 방향의 시작각 `θ₀=normal_offset_rad`도 고정 설정으로 저장한다. `B`개 블록의 현재 중심 `b_i`로부터 중심과 법선을 다음처럼 계산한다.

$$
c_{obj}=\frac1B\sum_{i=1}^{B}b_i,\qquad
\nu_k=\begin{bmatrix}
\cos(\theta_0+2\pi k/n)\\
\sin(\theta_0+2\pi k/n)
\end{bmatrix},\quad k=0,\ldots,n-1.
$$

각 변까지의 공통 거리(내접 반경)를

$$
r_{obj}=\max_{i,k}\nu_k^\top(b_i-c_{obj})+m_{obj}
$$

로 두고, 허용 영역을

$$
\mathcal P_{obj}=\{q:\nu_k^\top(q-c_{obj})\le r_{obj}\ \forall k\}
$$

로 정의한다. 따라서 계획 제약은

$$
\nu_k^\top C_h A
\le r_{obj}+\nu_k^\top(c_{obj}-q_0)
\quad\forall h,k.
$$

`r_obj>0`이면 균등 배치한 법선과 공통 거리로 정확히 n개의 변을 가진 정다각형이 된다. 모든 블록에 대해 `ν_kᵀ(b_i−c_obj)≤r_obj−m_obj`이므로 **모든 블록 중심이 포함됨을 구성 자체로 보장**한다. 최소 포괄 원을 먼저 계산하거나 원을 내부 근사하는 절차는 필요하지 않다. 중심은 블록 평균으로 고정하므로 모든 중심·회전을 탐색한 최소 면적 n각형이라는 주장은 하지 않는다.

`m_obj>0`는 각 변을 바깥으로 이동하는 수직 거리 margin이며 workspace의 안쪽 margin과 구분한다. 초기 후보는 `0.10 m`, `n=8`, `θ₀=0`으로 두고 태스크별 validation에서 선택한다. 블록이 겹치거나 공선이어도 양의 margin이면 정상적인 다각형을 만든다.

기본은 블록 **중심**을 포함한다. 전체 외형까지 포함하려면 물체별 외접 반경 `r_i`를 사용하여 `r_obj=max_(i,k)[ν_kᵀ(b_i−c_obj)+r_i]+m_obj`로 확장하는 옵션을 둔다. 이 경우 각 물체의 외접 원판까지 포함하는 보수적인 영역이다.

적용 규칙은 다음과 같다.

- 성공한 블록을 포함하여 모든 블록을 사용한다. 가장 가까운 블록 선택이나 성공 블록 제외는 기본에 넣지 않는다.
- 모델이 생성한 미래 블록 좌표가 아닌, 재계획 시점의 실제 관측으로 다각형을 만든다.
- 하나의 Predict→VLSA→PC 안에서는 중심·법선·내접 반경을 고정한다. 다음 재계획에서 새 관측으로 갱신한다.
- workspace와 다각형의 **교집합**을 사용한다. Object Envelope 자체의 물체 포함과, workspace를 교차한 최종 허용 영역의 물체 포함은 구분한다. workspace 밖의 블록 때문에 workspace를 넓히지 않는다.
- 현재 목표가 다각형 밖에 있어도 margin을 자동으로 확대하지 않는다. 가속도 제약과 함께 다음 목표가 영역에 들어갈 수 있는지를 공동 feasibility로 판정한다. 불가능하면 `infeasible`로 기록한다.

이 제약은 물체들에서 지나치게 멀리 벗어나는 것을 제한한다. 다각형 내부에서도 손이 모든 물체에서 멀리 있을 수 있으며, **각 물체를 골고루 다루는 행동이나 과제 성공을 보장하지 않는다.** 물체별 진척·성공률을 별도로 확인한다.

## 5. 공통 QP 표현

### 5.1. 가속도 L2 norm의 보수적인 선형 근사

사각형 workspace와 볼록 n각형 Object Envelope는 정확하게 선형 부등식으로 표현된다. 남아 있는 비선형 제약은 **가속도 L2 norm**이다. 이를 정확히 유지하면 최소 수정 문제는 볼록 QCQP/SOCP가 되므로, QP 통합에서는 다음 근사를 사용한다.

`m=constraints.acceleration.polygon_sides`를 짝수 `m≥4`로 정하고

$$
\mu_k=(\cos(2\pi k/m),\sin(2\pi k/m)),\qquad
\beta=\alpha_{max}\Delta t^2\cos(\pi/m)
$$

를 정의한다. 원판 `‖a_h−a_(h−1)‖₂≤α_max Δt²` 내부의 정m각형을

$$
\mu_k^\top(a_h-a_{h-1})\le\beta\quad\forall k
$$

로 표현한다. 짝수 면수를 사용하여 양·음의 가속도에 대해 중심 대칭이 되게 한다. 기본 후보 `m=32`에서는 내접 반경이 원판 반지름보다 약 `0.482%` 작다. 우변에서 `cos(π/m)`를 빼면 외접 다각형이 되어 원래 L2 한계를 넘는 가속도를 허용하므로 사용하지 않는다.

행동 계획 변수로 쓰면

$$
\mu_k^\top(S_h-S_{h-1})A\le\beta\quad(h\ge1),
\qquad
\mu_k^\top S_0A\le\beta+\mu_k^\top a_{prev}\quad(h=0).
$$

따라서 가속도 역시 QP의 선형 제약이 된다. Object Envelope의 `n_sides`와 가속도 근사의 `polygon_sides`는 **별도 설정**이다. 전자는 작업 영역의 모양을, 후자는 L2 가속도 제한의 근사 정확도를 결정한다.

### 5.2. 공통 행렬과 제약 의미

모든 면을 쌓으면

$$
\mathcal C_{QP}(\xi)=\{A:G(\xi)A\le d(\xi)\}
\subseteq\mathcal C_{L2}(\xi)
$$

가 된다. 오른쪽 집합은 같은 사각형·Object Envelope와 **정확한 L2 가속도 한계**를 쓴 집합이다. 포함 관계에서 발생하는 보수성은 가속도 근사뿐이며, Object Envelope에 별도 원 제약은 없다.

세 기본 제약만 켜면 계획 시점당 workspace 4면, Object Envelope n면, 가속도 m면이므로 총 `H(4+n+m)`개의 선형 부등식이다. 면마다 `constraint_id`, `horizon_index`, 기하 정보, 양의 장벽 정규화 길이 `s_j`를 저장한다.

가속도 면을 행동 차분 단위로 컴파일했으므로 모든 QP 행의 잔차는 m 단위다. 실제 가속도 L2 값·초과량은 별도로 `Δt²`로 나누어 m/s²로 보고한다. 원래 L2 조건은 만족하지만 내부 다각형 때문에 실패하는 경우를 구분한다.

### 5.3. 공통 최소 수정 문제

Projection과 계획 VLSA는 같은 문제를 다른 시점에 적용한다.

$$
A^*=\arg\min_A\frac12(A-A_{nom})^\top W(A-A_{nom})
\quad\text{s.t.}\quad GA\le d.
$$

기본 `W=I`이며 raw m 단위에서 비교한다. 초기 통합에서는 임의의 시간 가중치·관측 손실을 추가하지 않는다. 제약별 순차 clipping이 아니라 모든 활성 제약의 교집합을 한 번에 푼다.

새 다각형은 XY축을 결합하므로 기존 `BoxQP`의 축별 분리만으로 처리할 수 없다. **일반 선형 부등식과 양의 정부호 목적행렬을 받는 QP solver adapter**를 마련해야 한다. 특정 solver 패키지 선택은 구현 단계에서 정하되, solver status와 원래 물리 단위 잔차를 공통 기준으로 검증한다.

## 6. 세 단계의 동작 명세

### 6.1. Predict와 Projection

조건부 노이즈 `X₀`에서 `N_p`회 Euler 생성한다.

$$
Y_{k+1}=X_k+\frac1{N_p}v_\theta(X_k,k/N_p).
$$

Projection on이면 raw 행동 부분을 5.3절의 QP로 바꾼다. 수정한 행동을 정규화하여 관측 채널과 다시 결합하고 첫 관측을 재고정한다. off이면 Euler 결과를 그대로 사용한다. 초기 노이즈를 별도 투영하지 않는다.

`scope: predict`로 한정한다. Projection on이라는 이유로 PC 내부에 다시 Projection을 삽입하지 않는다. PC 이후 추가 투영은 알고리즘을 바꾸므로 별도 실험 옵션으로만 허용한다.

### 6.2. VLSA: 모델 없는 terminal 계획 보정

Predict가 반환한 행동 `A_pred`를 명목 계획으로 하여 5.3절의 QP를 한 번 푼다. 현재 VLSA의 한 행동 필터를 단순히 H번 순서대로 호출하지 않고, **누적 목표 제약을 포함한 전체 horizon을 공동 보정**한다.

이 변경은 `vlsa.scope: terminal_plan`으로 기록한다. 기존 `execution_step` 방식의 결과와 같은 알고리즘으로 취급하지 않는다. 또한 같은 집합·목적함수를 쓰면 통합 VLSA의 이 단계는 수학적으로 **Projection의 `final` 모드와 동일하다.** 두 이름을 별개의 새 최적화 알고리즘으로 해석하지 않는다.

따라서 `every`가 마지막 생성 스텝까지 성공적으로 투영했고 중간에 기하가 바뀌지 않았다면, 뒤의 VLSA는 이론적으로 같은 계획을 반환한다. 이 조합은 구조상 허용하지만 수치 허용 오차 밖의 변화가 크면 구현 불일치를 의심해야 한다.

### 6.3. PC Correct: 모델을 사용하는 보정

입력은 앞 단계의 최종 계획이며 새 노이즈를 생성하거나 Predict를 다시 수행하지 않는다. 보정 시간만 0으로 초기화하고 `N_c`회 반복한다.

$$
\widetilde v_\tau=\alpha(1-\tau)v_\theta(X^c_\tau,\tau).
$$

raw 행동 벡터장 `V`는 정규화 행동 벡터장에 행동 scale을 곱해 얻는다. 이는 `dA/dτ`이며 실제 제어 시간의 속도가 아니다.

공통 QP 면에서 정규화 장벽을 정의한다.

$$
h_j(A)=\frac{d_j-G_jA}{s_j},\qquad
\phi(h)=\varepsilon\operatorname{sgn}(h-\delta)|h-\delta|^\rho.
$$

`∇h_jᵀU+φ(h_j)≥0`은 다음 선형 부등식으로 변환된다.

$$
G_jU\le s_j\phi(h_j(A)).
$$

따라서 보정 QP와 갱신은

$$
U^*=\arg\min_U\frac12\|U-V\|_2^2
\quad\text{s.t.}\quad G_jU\le s_j\phi(h_j(A))\ \forall j,
\qquad A\leftarrow A+\Delta\tau U^*.
$$

여기서 `s_j`는 workspace 면에는 해당 축 반폭, Object Envelope 면에는 `r_obj`, 가속도 면에는 `β=α_max Δt² cos(π/m)`를 사용한다. 모두 m 단위의 양수다. 같은 면의 scale과 `a_prev`를 보정 도중 바꾸지 않는다. 가속도 면의 `G_j U`는 인접 행동 벡터장의 차분을 포함하며, `h=0`의 이전 실행 행동은 고정값이므로 생성 시간 미분이 0이다. 이 일반 행렬 방식으로 기존 box 전용 PC를 확장한다.

관측 채널은 스케일된 명목 벡터장으로 갱신하며 첫 관측만 고정한다. 보정된 행동에 맞춰 미래 관측을 임의로 덮어쓰지 않는다. 두 예측 사이의 불일치는 필요 시 별도 지표로 남긴다.

초기 후보는 기존의 `alpha=2`, `rho=0.9`, `delta=0.01`이다. 자동 `eps`는 선형 장벽에 대한 기존 식을 사용한다.

$$
b_*={\delta\over1-\rho},\qquad
\varepsilon={0.9\over\Delta\tau}{b_*\over(b_*-\delta)^\rho}.
$$

`0<rho<1`, `delta>0`를 검사한다. 이 설정은 feasible한 선형 장벽의 Euler 경계 통과를 막기 위한 조건이다. 서로 충돌하는 CBF 요구, QP 오차, 처음부터 infeasible한 계획의 유한 횟수 복구까지 보장하지 않는다. 매 갱신 잔차와 최종 feasible 여부를 검사하고 실패를 반환한다. **마지막에 숨은 Projection이나 VLSA를 붙이지 않는다.**

### 6.4. 실행 시점 VLSA 재검사

기본 3단계 비교에서는 `vlsa.execution_recheck=false`로 둔다. 기존 VLSA의 장점을 보존할 필요가 있으면 별도 모드로 활성화한다.

이 모드에서는 각 행동 직전에 최신 관측으로 `ξ`를 재구성하고, 현재 목표 `q_t`, 직전 실제 실행 행동과 큐의 행동에 대해 `H=1` 최소 수정 문제를 푼다. `vlsa.enabled=false`이면 실행하지 않는다. 수정된 목표를 다음 관측에 반영한다.

행동이 수정되면 나머지 큐의 누적 목표는 원래 계획과 달라진다. 따라서 **수정이 허용 오차를 넘으면 그 행동 실행 뒤 남은 큐를 폐기하고 재계획**한다. 재검사 off에서는 계획 생성 당시의 고정된 기하만 보장 대상이며, 움직인 블록을 기준으로 한 최신 Object Envelope 제약은 보장하지 않는다.

## 7. 독립적인 on/off와 호출 예산

기본 플래그는 `methods.projection.enabled`, `methods.vlsa.enabled`, `methods.pc.enabled`다. 순서는 config로 재배열하지 않고 고정한다.

| Projection | VLSA | PC | 경로 |
|---|---|---|---|
| off | off | off | Euler → 실행 |
| on | off | off | 투영 Euler → 실행 |
| off | on | off | Euler → terminal QP → 실행 |
| off | off | on | Euler → PC Correct → 실행 |
| on | on | off | 투영 Euler → terminal QP → 실행 |
| on | off | on | 투영 Euler → PC Correct → 실행 |
| off | on | on | Euler → terminal QP → PC Correct → 실행 |
| on | on | on | 투영 Euler → terminal QP → PC Correct → 실행 |

### 7.1. 기본 비교: 생성 품질 유지

`predict_steps=10`, PC on일 때 `correct_steps=9`로 정한다. 따라서 PC off의 신경망 호출 수는 10회, on은 **19회**다. 플래그를 켰다고 앞 단계의 Euler 횟수를 암묵적으로 줄이지 않는다.

이 설정은 기존 PC의 1+9회와 다르다. 순수한 모델 호출 수가 같은 비교라고 주장하지 않고 실제 호출 수·지연을 함께 보고한다.

### 7.2. 추가 비교: 동일 모델 호출 예산

총 호출 예산 10회 비교에서는 PC on에 `predict_steps=1`, `correct_steps=9`, PC off에 `predict_steps=10`을 명시한다. 이는 별도 설정 묶음이다. 이때 PC on의 Projection은 **생성 갱신이 한 번뿐이므로 여러 noisy intermediate를 보정하는 효과를 볼 수 없다.** `5+5` 같은 분할도 별도 ablation으로 둔다.

샘플러 구조·제약 집합이 달라지므로 통합 1+9 결과를 기존 PC의 완전 재현으로 부르지 않는다. 기존 코드·기존 집합으로 재현하는 legacy 비교와 통합 비교를 분리한다.

## 8. 설정 구조

아래는 **제안 스키마 예시**이며 현재 `configs/config.py`에서 바로 실행되는 YAML은 아니다. 기존 경로·모델·평가 설정에 이 구조를 추가한다. 설정 버전이 다르면 묵시적 변환 없이 명시적으로 마이그레이션한다.

```yaml
schema_version: 2
control:
  dt_s: 0.035
task: pushing
sampling:
  predict_steps: 10
methods:
  projection:
    enabled: false
    mode: every
    scope: predict
    skip_first: 0
  vlsa:
    enabled: false
    scope: terminal_plan
    execution_recheck: false
    invalidate_queue_on_change: true
  pc:
    enabled: false
    correct_steps: 9
    alpha: 2.0
    rho: 0.9
    delta: 0.01
    eps: auto
constraints:
  workspace:
    enabled: true
    geometry: rectangle
    source: research_bounds
    low_m: [0.20, -0.40]
    high_m: [0.85, 0.50]
    margin_m: 0.01
  acceleration:
    enabled: true
    max_m_s2: null       # 태스크별 학습 데이터로 정한 양의 값을 입력해야 함
    approximation: inscribed_polygon
    polygon_sides: 32
  object_envelope:
    enabled: true
    geometry: regular_polygon
    n_sides: 8
    normal_offset_rad: 0.0
    center: mean_blocks
    include: all_blocks
    footprint: centers
    margin_m: 0.10
    update: per_plan
    auto_expand: false
solver:
  formulation: linear_inequality_qp
  dtype: float64
  feasibility_tolerance_m: 1.0e-7
  intervention_tolerance_m: 1.0e-7
  on_failure: abort_episode
```

workspace는 `geometry: rectangle`과 `low_m/high_m`만 허용한다. Object Envelope의 `n_sides`는 태스크별로 변경할 수 있다. 이 예시는 한계값 선정 전의 템플릿이며, 활성 가속도의 `max_m_s2: null`을 양의 실수로 채우기 전에는 검증을 통과하지 않는다. Inserting YAML은 `task`, 경로·모델·평가 설정과 workspace 경계를 바꾸며 나머지 제약 수치도 독립적으로 지정할 수 있다.

검증 규칙은 다음과 같다.

- CLI > YAML > 명시한 공통 기본값 순서이며 최종 `resolved_config.yaml`을 저장한다.
- 알 수 없는 키·잘못된 타입·지원하지 않는 제약을 즉시 오류로 처리한다. 기존 PC처럼 `step_norm`을 조용히 무시하지 않는다.
- 활성 방법에 적용할 제약 목록은 공통으로 하나다. 방법별 별도 목록은 기본 스키마에서 금지한다.
- 모든 방법 off면 제약은 측정용으로만 사용한다. 제약 위반 때문에 baseline 행동을 거부하지 않는다.
- `N_p≥1`, 활성 PC의 `N_c≥1`, `1≤execute_steps≤H`, Object Envelope의 정수 `n_sides≥3`, 가속도 근사의 짝수 정수 `polygon_sides≥4`를 검사한다.
- workspace margin 적용 후 양의 폭, Object Envelope의 양의 margin, 유한한 시작각, 양의 `dt_s`, `max_m_s2`와 장벽 scale을 요구한다. 활성 옵션의 `null` 값은 오류다. 실제 제어 간격과 `dt_s`가 일치하는지 확인한다.
- `projection.mode=final`은 비교 전용이다. 동일 terminal QP를 중복 실행하는 `vlsa.scope=terminal_plan`과 함께 켜면 설정 오류로 처리한다.
- 단일 스텝 Predict를 지원하도록 기존 `len(t_span)>1` 가정을 수정해야 한다. baseline의 다중 스텝 경로 동등성은 유지한다.

QP 잔차 허용 오차는 raw 거리(m)로 검증한다. 가속도는 `max(‖a_h−a_(h−1)‖−α_max Δt²,0)`도 m 단위로 검사하고, 이를 `Δt²`로 나눈 초과량을 m/s²로 보고한다. 따라서 같은 `1e−7`을 m와 m/s²에 그대로 재사용하지 않는다. 모델 float32 복귀 오차를 고려한 초기 거리 기준은 `1e−7 m`이며, solver 내부 중단 기준은 그보다 엄격하게 정한다.

## 9. 공통 구조와 모듈 책임

통합의 소유 패키지는 `fm_constraint/`다. `PLAN.md`에 따라 baseline과 비슷한 평평한 구조로 구현한다. 세분화한 하위 패키지와 단계별 factory는 두지 않는다.

```text
fm_constraint/
  train.py                  # baseline 학습 위임
  sample.py                 # baseline 데이터 도구 + 통합 정책
  eval_pushing.py            # Pushing 평가 진입점
  eval_inserting.py         # Inserting 평가 진입점
  policy.py                 # Predict → Projection → VLSA → PC 전체 순서
  constraints.py            # 세 제약의 공통 행렬·장벽 scale·위반 검사
  qp.py                     # OSQP 기반 공통 선형 부등식 QP
  evaluation.py             # 공통 실행·이전 행동·큐·로그
  check_constraints.py      # 수치 검사·학습 통계
  compute_metrics.py        # 결과 집계·context bootstrap
  render_rollouts.py        # 저장 궤적 지도·카메라 PNG/GIF
```

논리적 API는 다음으로 통일한다.

```python
context = build_context(observation, task_geometry, previous_executed_action, dt_s)
constraints = compile_constraints(context, horizon, constraint_config)
plan = predict(model, noise, observation, projection=optional_projector)
plan = terminal_correct(plan, constraints) if vlsa_enabled else plan
plan = pc_correct(model, plan, observation, constraints) if pc_enabled else plan
report = validate_plan(plan, constraints)
```

각 보정은 `value`, `status`, `max_residual`, `intervention_norm`, `iterations`, `latency_ms`를 반환한다. `constraints`는 한 계획의 모든 단계가 같은 객체·기하를 공유한다. 모델 호출 계측은 solver 호출과 분리한다.

`fm_baseline`의 모델 로딩·정규화·데이터 로더·관측 구성·학습을 재사용한다. 새 학습, terminal 데이터 추가, padding, checkpoint 재저장을 통합의 전제로 삼지 않는다. `fm_pc`, `fm_projection`은 이관·검증 중에만 유지한다. 결과·설정·Git 커밋을 보존하고 PLAN의 삭제 조건을 충족하면 제거한다. 통합 코드는 구 패키지를 import하지 않으며 삭제 후 호환 wrapper도 남기지 않는다.

오프라인 `sample`도 통합 `plan()`을 호출하도록 확장해야 한다. 현재처럼 baseline sample만 호출하면 새 방법의 오프라인 검사가 불가능하다. `train`은 기존 학습을 위임하며 추론 보정 설정을 학습 손실에 반영하지 않는다.

## 10. 실패 처리와 최종 검사

불가능한 제약을 숨기기 위해 경계를 clip하거나 다른 방법을 자동으로 켜지 않는다.

| 상태 | 처리 |
|---|---|
| `invalid_input` | NaN/Inf·shape·설정 오류. 해당 행동 실행 안 함 |
| `infeasible` | 원래 선택한 QP 집합에 해 없음. episode 실패로 기록 |
| `solver_failure` | 미수렴·수치 오류·시간 한도. 마지막 iterate를 성공 해로 사용하지 않음 |
| `constraint_violation` | 활성 보정 파이프라인의 최종 결과가 허용 오차를 초과. 실행 안 함 |
| `ok` | 반환 dtype까지 변환한 결과의 잔차 검사 통과 |

활성 보정 방법이 있는 실험에서는 최종 검사 실패를 episode 실패로 포함한다. 실패 episode를 결과에서 제외하지 않는다. 모든 방법 off인 baseline에서는 같은 위반을 **측정만** 하고 baseline 그대로 실행한다. 이 차이가 성공률에 미치는 영향을 볼 수 있도록 `filter_abort_rate`를 별도로 보고한다.

현재 목표 `q₀`가 workspace 밖에 있는 초기 상태는 별도 `initial_outside_workspace`로 기록하고, 통합 기본 보정 모드에서는 실행을 중단한다. Object Envelope는 재계획 때 움직일 수 있으므로 현재 목표가 밖에 있다는 이유만으로 중단하지 않고, 다음 목표를 만족시키는 공동 해의 존재로 판단한다.

정지 행동도 현재 목표가 영역 밖이면 feasible하지 않을 수 있다. 실패 시 무조건 0 행동을 실행하지 않는다. slack·복구·CBF 완화 정책이 필요하면 별도 명시적 설정과 ablation으로 추가한다.

## 11. 데이터셋과 모델의 공통 기준

현재 저장된 baseline 설정과 전처리 데이터를 사용한다. 통합 때문에 데이터 split·전처리·모델 구조를 변경하지 않는다.

| 항목 | Pushing | Inserting |
|---|---|---|
| 물체 수 | 2 | 3 |
| 관측 차원 | 12 | 16 |
| 행동 차원 | 2: 목표 TCP Δx, Δy | 동일 |
| 상태·행동 순서 | `[action, obs]` | 동일 |
| horizon | 16 | 16 |
| 학습/검증 시연 | 1,800 / 200 | 718 / 82 |
| 길이 16 학습/검증 window | 390,518 / 42,868 | 701,138 / 79,249 |
| 분할 | 공식 train/eval 파일 | context 단위 자체 분할 |
| 모델 | Temporal U-Net | Temporal U-Net |
| `dim`, `dim_mults` | `32`, `[1,4,8]` | `64`, `[1,2,4,8]` |
| attention / spectral norm | 없음 / 없음 | 없음 / 없음 |
| 학습 | CFM, 100만 update, batch 32 | 동일 |
| 정규화 | 학습 채널별 center/scale | 동일 방식, 별도 통계 |
| 평가 가중치 | 기존 `baseline_ema.pt` | 기존 `baseline_ema.pt` |
| 제어 기록 간격 | 0.035 s | 0.035 s |

출처는 각 태스크의 [`config.json`](../fm_baseline/results/pushing_baseline/config.json), [Inserting 설정](../fm_baseline/results/inserting_baseline/config.json), [`data/`](../data/)의 manifest다. checkpoint와 normalizer의 hash를 저장하여 같은 모델인지 확인한다.

평가는 Pushing의 고정 test context 목록, Inserting의 고정 validation context 목록을 방법 간 공유한다. 통합 기본 실행 묶음은 `execute_steps=8`, 최대 실행 길이는 Pushing 400, Inserting 2,000으로 맞춘다. Inserting validation으로 margin 등을 조정했다면 그 결과는 validation 성능으로 표시하며 독립 test 성능으로 부르지 않는다.

## 12. 실험·검증 방침

### 12.1. 비교군

먼저 동일 제약·동일 checkpoint로 7절의 8개 on/off 조합을 수행한다. 실행 재검사는 모두 off로 고정한다. 이후 `execution_recheck=true`, 생성/보정 호출 예산, 제약별 제거 실험을 별도 비교한다.

같은 episode/context/replan index에서 같은 초기 노이즈를 사용할 수 있도록 RNG를 명시적으로 관리한다. 각 방법이 다른 실행 경로를 만들면 실제 관측까지 같아지는 것은 아니다. 단일 관측의 오프라인 비교와 closed-loop 비교를 분리한다.

### 12.2. 기록할 값

| 분류 | 필수 기록 |
|---|---|
| 과제 | 성공률, 모드 엔트로피, D3IL score, 물체별 성공·최종 거리 |
| 제약 | 사각형·물체 다각형 위반(m), 가속도 L2 값·초과량(m/s²), QP 면 잔차(m), 목표/실제 손 지표 구분 |
| 단계 | Predict 종료·VLSA 종료·PC 종료의 행동과 위반량·수정량 |
| 기하 | 재계획별 물체 좌표, 다각형 중심·법선·내접 반경·n·margin, 가속도 근사의 m·한계·제어 간격 |
| 비용 | 모델 호출 수, QP 호출 수, 단계별/전체 지연, 재계획 횟수 |
| 실패 | infeasible, solver failure, 최종 위반, 초기 영역 밖, 중단 비율 |
| 실행 | 계획 ID, 실행 행동, 현재/다음 목표, 최신 실제 손, 큐 폐기 여부 |
| 재현 | resolved config, 코드 버전, checkpoint/normalizer hash, seed |

실제 접촉 계측이 없으면 물리적 `safe_success_rate`는 `null`로 둔다. 목표 영역 안에 있었다는 지표와 혼용하지 않는다. 위반율·개입률은 분모가 계획/행동 시도/실제 실행 중 무엇인지 명시한다. 신뢰구간은 같은 context를 묶어 비교한다.

### 12.3. 구현 완료 판정

1. 모든 방법 off에서 같은 noise·관측·checkpoint에 대한 baseline 계획이 일치하고, 같은 모의 환경에서 실행 순서가 일치한다. 실제 D3IL에서도 별도 확인한다.
2. 세 방법이 동일 context에서 동일한 `G,d`와 원래 제약 정의를 사용한다. 원점·단위·부등식 방향·물체 누락을 검사한다.
3. Object Envelope의 n개 반공간이 모든 블록 중심을 포함하고 정n각형을 이루는지 확인한다. 중복·공선 블록과 재계획에 따른 영역 이동을 검사한다. 가속도 근사의 꼭짓점은 정확한 L2 가속도 원판 안에 있어야 한다.
4. QP 해를 독립적인 수치 기준과 비교하고, feasible/infeasible·경계·모델 dtype 복귀를 확인한다.
5. VLSA 보정의 멱등성, `Projection final`과의 동일성, `every` 뒤 VLSA의 중복성을 확인한다.
6. PC가 공통 가속도 면까지 반영하고, 첫 관측을 유지하며, 예상한 모델 호출 횟수만 사용하는지 확인한다. 첫 계획·재계획 경계·오프라인 시연 경계에서 올바른 이전 행동과 `Δt²`를 사용하는지도 검증한다.
7. 동적 Tracking의 계획 중 고정·재계획 시 갱신, 실행 재검사 후 큐 폐기를 확인한다.
8. 실패 결과가 실행되지 않고 모든 실패가 통계에 포함되는지 확인한다.
9. 두 태스크에서 같은 평가 조건으로 8개 조합의 최소 실행 검증을 통과한 뒤 본 실험을 수행한다.

## 13. 구현 순서

1. **공통 제약부터 구현:** context, 물체 포괄 정n각형, 가속도 차분·내접 근사, 위반 함수, 일반 QP adapter를 만든다.
2. **통합 정책에 연결:** 공통 Predict와 Projection, terminal VLSA, PC Correct를 순서대로 연결한다. 기존 PC의 중복 Predict 호출을 제거한다.
3. **설정·실행을 통일:** 새 schema와 공통 평가 루프를 연결하고 실패·단계 로그를 추가한다. 실행 재검사는 기본 off로 유지한다.
4. **재현과 ablation:** legacy 결과를 보존한 상태에서 off 동등성, 8개 조합, 동일 호출 예산 비교를 수행한다.

이 문서의 초기 수치 중 workspace는 기존 연구용 범위를 계승하고, `object_envelope.margin_m=0.10`, 물체 다각형 8면, 가속도 근사 32면과 공통 허용 오차는 새 통합의 초기 후보값이다. 물리적 도달 가능성 검증, 태스크별 가속도 한계·margin 선택, QP solver 패키지 선정은 후속 구현·검증 항목으로 남기되, 파이프라인 순서·제약 의미·실패 처리·비교 원칙은 위 명세를 따른다.


## 14. 구현에 반영한 초기 설정값

2026-10-24 구현에서는 원래 후보값 대신 학습 데이터에서 산정한 고정 설정을 사용한다.

| 항목 | Pushing | Inserting | 근거 |
|---|---|---|---|
| 가속도 한계 | 13.7 m/s² | 14.2 m/s² | 학습 시연의 시작 행동 포함, 다각형 필요 한계 최대값 +5% |
| 물체 margin | 0.25 m | 0.14 m | 모든 학습 시연의 초기 목표를 포함하는 필요 margin +1cm, 올림 |
| 물체 면수 / 가속도 근사 면수 | 8 / 32 | 8 / 32 | 태스크별 config에서 독립 설정 |

원래 공통 margin 0.10m는 Pushing의 일부 시작 목표를 제외하여 가속도 제한과 공동 적용하면 infeasible했다. 현재 값은 실행 중 자동 확대하지 않는다. 수치 근거와 학습 데이터 hash는 `fm_constraint/results/*_calibration.json`, 재산정 명령은 `fm_constraint.check_constraints`에 있다. 이 표는 앞 절의 초기 후보 수치를 대체하며, 상세 구현·검증 상태는 PLAN의 진행 현황을 따른다.
