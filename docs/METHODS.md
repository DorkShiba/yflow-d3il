# Train-free 5개 방법: 현재 구현 명세

이 문서는 `fm_constraint/` 코드가 실제로 하는 일을 기준으로 쓴다(schema v4, 2026-11-08). 실행 방법은 [`fm_constraint/README.md`](../fm_constraint/README.md), 결정 이력은 [`PLAN.md`](PLAN.md)를 본다.

## 0. 공통 설정

| 기호 | 값 | 의미 |
|---|---|---|
| H | 16 | 계획 길이 |
| s | 8 | 한 계획에서 실행하는 행동 수(이후 재계획) |
| N | 10 | Predict의 Euler 스텝 수 |
| Δt | 0.035 s | 제어 간격 |
| a_h | ℝ² | 행동 = 목표 TCP XY 증분(m) |
| q₀ | ℝ² | 계획 시점의 현재 목표 TCP XY |

모델은 정규화 공간에서 `[a(2), obs]` 궤적 H개를 생성하고, 첫 시점 관측은 매 Euler 뒤 실제 관측으로 다시 고정한다. 행동 h의 목표 위치는 `q_h = q₀ + Σ_{i≤h} a_i`다.

```text
noise → Predict N회 [ FM 또는 ④ RTC 벡터장 + ② PF2MP → Euler → ① Projection ]
      → ⑤ LiPo → 큐(s개) → 행동마다 ③ Filter → env.step
```

순서는 고정이며 설정으로 바꾸지 않는다. 결과에 `pipeline_order: [rtc, pf2mp, projection, lipo, filter]`로 기록한다.

## 1. 명시 제약 집합 C (①, ③, ⑤가 공유)

목표 명령에 거는 선형 부등식 `G a ≤ d`이며, 재계획 시점의 관측으로 한 번 만들고 한 계획 동안 고정한다(`fm_constraint/constraints.py`).

| 제약 | 정의 | Pushing | Inserting |
|---|---|---|---|
| Workspace | `low+m ≤ q_h ≤ high−m` | `[0.21,0.84]×[−0.39,0.49]` m | `[0.01,0.84]×[−0.39,0.74]` m |
| 가속도 | `‖a_h − a_{h−1}‖/Δt² ≤ A`, 32면 내접 다각형. `a_{−1}` = 직전 **실제 실행** 행동 | A = 13.7 m/s² | A = 14.2 m/s² |
| Object envelope | 모든 블록 중심을 포함하는 정8각형 + margin, 중심은 블록 평균 | margin 0.25 m | margin 0.14 m |

수치는 train 시연 통계로 정했다(`fm_constraint/results/*_calibration.json`). 실제 TCP·링크·블록 충돌을 보장하는 제약은 아니다. 제약 위반율은 실제 실행 행동과 **실제 상태**(OOD 관측 노이즈 제외)로 측정한다.

## 2. 방법별 명세

### ① Projection (`methods.projection`, 비트 16)
- `mode: every`: 매 Euler 직후 행동 H개를 C에 공동 투영한다. `min ‖a − a_nom‖² s.t. G a ≤ d`, OSQP, float64. `skip_first`로 앞 스텝을 건너뛸 수 있다.
- `mode: final`: Predict 끝에 한 번만 투영한다. 이전의 terminal-plan VLSA와 같은 연산이다.
- 켜져 있으면 최종 계획이 C를 만족해야 한다. 투영이 실패하거나 잔차가 1e-7 m를 넘으면 계획을 실행하지 않고 episode를 실패로 기록한다.

### ② PF2MP (`methods.pf2mp`, 비트 8)
- train 시연의 목표 TCP XY로 만든 Gaussian KDE와 고밀도 영역 거리장으로 potential `φ(q) = log p(q) + α·dist(q, safe)`를 정의한다.
- Predict 벡터장에 `λ∇φ`를 **생성된 미래 상태 위치 채널에만** 더한다. 행동 채널 가산항은 항상 0이고, 행동은 다음 FM 평가를 거쳐 간접적으로 바뀐다.
- 기본값(λ 0.01, bandwidth 0.025 m, α −1)은 초기 후보값이다. 구현은 `fm_constraint/pf2mp.py`에 있다.

### ③ Filter: 실행 직전 필터 (`methods.filter`, 비트 4)
- 큐에서 꺼낸 행동 하나를, 실행 직전 **최신 관측**(블록 위치, 직전 실제 행동)으로 만든 C(H=1)에 투영한다(OSQP).
- 행동이 1e-7 m보다 크게 수정되면 남은 큐와 대기 중인 계획(지연 E)을 버리고 다음 step에서 재계획한다.
- 이 저장소의 원래 VLSA 정의다. ①과 함께 켜도 둘 다 실행한다. ①은 계획 시점의 기하로 H개를 다루고, ③은 실행 시점의 기하로 1개를 다룬다.

### ④ RTC (`methods.rtc`, 비트 2)
- Real-Time Chunking(식 2–5)의 행동 inpainting guidance. 활성 계획에서 아직 실행하지 않은 H−s개 행동을 target으로 둔다.
- 매 Euler에서 `â = (x + (1−τ)v)[행동]`의 target 오차에 대한 입력 VJP를 더한다. soft mask W는 다음과 같고, 강도는 `min(β, (τ²+(1−τ)²)/(τ(1−τ)))`, β = 5다.
  - 앞 d개: 1 (지연으로 실행이 확정된 행동)
  - d ≤ i < H−s: `c(e^c−1)/(e−1)`로 감소
  - 이후: 0
- 첫 계획, 그리고 ③이 행동을 수정해 큐를 버린 직후에는 참조가 없어 적용하지 않는다(`rtc_guided: false`).
- 추론 지연은 OOD 조건 E의 d step으로만 모사한다(직렬 시뮬레이터, 벽시계 비동기 아님). 지연이 없으면 d = 0이다.

### ⑤ LiPo (`methods.lipo`, 비트 1)
- 계획 끝에서 고정 한도 minimum-jerk QP를 푼다(codex `mode="qp"` 정의, blend 없음):

  ```text
  min_z ‖D³ [history ; q₀ + L z]‖²
  s.t.  z_i = a_i                 (지연으로 확정된 앞 d개)
        |L(z − a)| ≤ bound        (각 시점·축의 위치 보정, 기본 0.3 mm)
        G z ≤ d                   (①이 켜져 있을 때만)
  ```

  history는 실행된 목표 위치 최근 3개(마지막 = q₀)이고, L은 행동 누적이다.
- ①이 켜져 있으면 ① 투영 결과가 이미 `G a ≤ d`를 만족하므로 z = a가 항상 해다. 보정 한도로는 닿을 수 없는 면(여유 > ‖G_row‖₁·2·bound)은 해를 바꾸지 않으므로 빼고 풀며, 최종 판정은 전체 면으로 한다.
- Clarabel(interior point)로 mm 단위에서 푼다. OSQP는 이 문제에서 수렴하지 않았다(PLAN 3단계 결과).
- baseline 평가 스크립트의 `--postprocess lipo`도 같은 함수(`fm_constraint/lipo.jerk_qp`)를 ① 제약 없이 쓴다.

## 3. 방법 간 상호작용 규칙

| 상황 | 규칙 |
|---|---|
| ①+⑤ | ⑤ QP에 ① 제약을 넣어 한 번에 푼다. 최종 계획은 C를 만족한다 |
| ③ 개입 | 큐·대기 계획 폐기 → 다음 계획은 ④ 참조 없음 |
| 지연 E (d step) | 남은 행동이 d개일 때 계획하고, 새 chunk는 index d부터 실행한다. 그 d개는 ④ mask 1, ⑤ 등식 고정 |
| ① 끔 | 계획 단계에서는 제약을 측정만 하고 거부하지 않는다 |
| 계획 단계 방법 모두 끔(①②④⑤), 고정 noise 없음 | baseline `Policy.plan`에 그대로 위임한다(추가 난수 없음) |

## 4. 조합과 비교

- 조합 ID = ①16 + ②8 + ③4 + ④2 + ⑤1 (`P00`~`P31`, `I00`~`I31`).
- Complementarity Δ = S_{A+B} − max(S_A, S_B). 비교 쌍은 실행 전에 고정했다. B = 조합에서 가장 낮은 비트의 방법, A = 나머지다(예: 28 = (24, 4), 31 = (30, 1)). 낮을수록 좋은 지표는 S = −x로 방향을 맞춘다.

## 5. 출처

| 방법 | 논문 | 이 저장소의 원래 구현 |
|---|---|---|
| ① | DPCC (L4DC 2025), SafeDiffuser (ICLR 2025) | `feature/method1` |
| ② | Towards Safe Imitation Learning via Potential Field-Guided Flow Matching (arXiv:2508.08707) | `feature/method123_merge` |
| ③ | AEGIS / VLSA (arXiv 2025) | `feature/method3` |
| ④ | Real-Time Execution of Action Chunking Flow Policies | `codex/rtc-qp-method123` |
| ⑤ | LiPo (arXiv:2506.05165) 위치 QP | `codex/rtc-qp-method123` (`mode="qp"`) |

PC(Predict-then-Correct, `feature/method2`), ACG, L2C2는 폐기했다. 기록은 git 이력에 있다.
