# RTC + QP 실험

기존 EMA 체크포인트를 고정하고 `baseline / QP / RTC / RTC+QP`를 비교한다.
RTC는 논문 *Real-Time Execution of Action Chunking Flow Policies*의 식 (2)–(5)를
사용한다. QP는 기존 `lipo.minimum_jerk`를 재사용하되 **LiPo blending은 적용하지 않는다**.

## 구현 범위

- `rtc.py::guided_plan`: 매 Euler 스텝에서 최종 행동 추정치
  `a_hat = (x + (1-tau) v)[..., :2]`를 만들고, 이전 청크의 미실행 행동에
  가까워지도록 VJP guidance를 더한다. 마스크는 frozen prefix 1, 중간 구간
  `c * (exp(c)-1)/(e-1)`, 새 구간 0이다. `beta=5`로 가중치를 제한하며 tau=0을 별도 처리한다.
- 모델은 `[ΔXY, observations]`를 공동 생성한다. 일치 오차는 **정규화된 행동 2채널**에만
  걸고 전체 입력에 대한 Jacobian은 유지한다. t=0의 관측 채널은 항상 현재 관측으로 고정한다.
  가중치는 학습하지 않으며 입력 gradient만 계산한다. attention은 필요 없다.
- 이전 참조는 **QP까지 적용해 실행하기로 한 행동 청크**다. 매 에피소드마다 상태를 초기화한다.
- `--postprocess qp`: 현재 목표 손 XY에서 ΔXY를 누적하고, 최근 실행 목표 위치 최대 3개를
  고정한 채 `||D3 [history; future + correction]||²`를 최소화한다. 결과를 다시 ΔXY로 바꾼다.
  `--lipo-path-bound-m`은 RTC가 생성한 위치 경로 대비 축별 보정 한도(m)이며 기본 0.0003이다.
  기존 `--lipo-blend-*` 옵션은 qp 모드에 영향이 없다. 충돌·동역학 제약은 이 QP에 없다.

## 추론 지연과 실행

**시뮬레이터 스텝 기반 지연 재현**이다. 실제 GPU inference와 env.step을 별도 스레드로
동시 실행하는 벽시계 비동기 제어기는 아니다. CPU/GPU 계산 시간을 제어 지연으로 자동 환산하지 않는다.

`H=16`, `s=8`이 기본이며 `0 <= d <= s <= H-d`를 검사한다.

1. 첫 청크는 에피소드 시작 전에 생성하며 RTC 참조는 없다. QP는 적용 가능하다.
2. 이후 s스텝 간격으로 최신 관측에서 새 청크를 생성한다. 기존 청크의 H-s개 행동이 참조다.
3. 결과는 d스텝 뒤에 활성화한다. 그동안은 기존 청크를 실행한다.
4. 새 청크의 앞 d개 행동을 기존 실행 예정 행동과 정확하게 일치시킨다. QP에서도 해당 위치의
   보정량을 0으로 고정한다. 활성화 시 이미 지나간 d개 행동은 다시 실행하지 않는다.
5. `d=0`이면 즉시 전환한다. 이 조건은 동기식 RTC guidance 실험이다.

RTC를 끈 지연 대조군도 같은 스케줄로 실행한다. delayed/RTC 경로에서는 기존
`blend/lipo` 모드를 거부해 혼합 방식과 시간 정렬의 혼선을 방지한다.
RTC의 연속성은 guidance로 유도하는 것이며 엄밀한 무충돌·jerk 상한 보장이 아니다.

## 실행

기존 D3IL/HardFlow 환경과 `requirements-lipo.txt`가 필요하다. 아래 명령은 저장소 루트에서 실행한다.
실행 결과는 기본 `codex/rtc_runs/`에 저장한다.

```bash
# 작은 4군 비교: 태스크별 3 context × 2회
python fm_baseline/run_rtc_comparison.py --task pushing --device cuda:0
python fm_baseline/run_rtc_comparison.py --task inserting --device cuda:0

# 모든 비교군에 같은 2스텝 추론 지연 적용
python fm_baseline/run_rtc_comparison.py --task pushing --device cuda:0 --inference-delay-steps 2

# 전체 Pushing 비교
python fm_baseline/run_rtc_comparison.py --task pushing --device cuda:0 --n-contexts 30 --n-traj 4

# 전체 Inserting 비교 (82개 검증 context)
python fm_baseline/run_rtc_comparison.py --task inserting --device cuda:0 --n-contexts 82 --n-traj 4

# 실행 명령만 확인, 파일 생성 없음
python fm_baseline/run_rtc_comparison.py --task pushing --dry-run
```

RTC+QP만 실행하려면:

```bash
python fm_baseline/eval_pushing.py \
  --run-dir fm_baseline/results/pushing_baseline --ckpt baseline_ema.pt \
  --execute-steps 8 --ode-steps 10 --n-contexts 3 --n-traj 2 \
  --rtc --rtc-beta 5 --postprocess qp --lipo-path-bound-m 0.0003 \
  --inference-delay-steps 0 --paired-noise --device cuda:0 \
  --output-root codex/rtc_single
```

`eval_inserting.py`도 같은 옵션을 지원한다. 재학습은 필요 없다. 보정 한도와 beta는
검증 데이터에서 선택하고 테스트 평가에서는 고정한다. 다른 제약 1·2·3의 통합은 포함하지 않는다.

## 평가와 로그

성공률, 실제/목표 손 jerk, 경계 jerk, 경계 속도 변화, 추종 오차와 실행 시간을 기존
smoothness 도구로 기록한다. 비교 실행기는 같은 체크포인트·노이즈·context·지연 조건의
네 군을 생성하고 baseline 대비 paired bootstrap 결과를 저장한다.

`results.json`에 RTC 사용 여부, beta, 지연 스텝, 실행/초기화 프로토콜을 기록한다.
`smoothness.json`의 각 plan에는 추론 시작·활성화 스텝과 frozen prefix 길이가 남는다.
`plan_starts`는 추론 시작이 아닌 **실행 청크 전환** 위치를 뜻한다.
추론 시간과 QP 시간을 별도로 기록하며 실제 벽시계 실시간 성능을 입증하는 값으로 해석하지 않는다.

기존 성공률 표와 바로 비교하지 말고 이 실행기의 baseline을 사용한다. 지연이 다른 결과는
paired 비교 도구가 거부한다. OOD 상태 생성은 이 변경에 포함하지 않았다.

## 테스트

```bash
python -m unittest discover -s tests -p 'test_rtc.py' -v
python -m unittest discover -s tests -p 'test_lipo*.py' -v
```

수식의 선형 모델 VJP 검증, 현재 관측 고정, beta=0 baseline 동일성, QP 한도·고정 prefix,
지연 0/1/4/8의 실행 정렬, QP 결과의 다음 청크 참조, 두 평가 루프의 모드·로그 연결을 검증한다.
가짜 환경 테스트는 실제 D3IL 성공률 검증을 대신하지 않는다.
