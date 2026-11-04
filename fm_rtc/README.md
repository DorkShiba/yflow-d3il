# Method 4: RTC (Real-Time Chunking)

기존 **순수 FM baseline EMA**에 RTC를 추론 시 적용한다. 추가 학습은 없으며
`fm_l2c2`의 학습 결과를 사용하지 않는다. 논문은 *Real-Time Execution of Action
Chunking Flow Policies*, arXiv:2506.07339v2 (NeurIPS 2025)이다.

## 구현

- `policy.py`: 논문 식 (2)--(5)의 최종 샘플 추정, vector-Jacobian product,
  soft mask, guidance clipping(beta=5), Euler 적분.
- D3IL 모델은 `[action, observation]`을 함께 생성한다. 마스크는 **정규화된
  행동 2개 채널만** 대상으로 한다. VJP는 자유로운 전체 샘플 좌표로 전달하지만
  첫 시점의 관측은 매 적분 단계 고정한다. 미래 관측을 이전 예측에 강제하지 않는다.
- 이전 청크를 새 추론 시작까지 경과한 `s`스텝만큼 이동해 target을 만든다.
  `d`개 앞부분은 가중치 1, 남은 overlap은 식 (5)대로 감소, 새 미래는 0이다.
  `d=0`에서도 overlap soft guidance는 작동한다.
- 매 Euler step에서 입력 샘플에만 미분한다. 파라미터 gradient와 optimizer는 없다.
  첫 청크, RTC off, beta=0에서는 기존 baseline Euler와 같은 계산을 한다.
- `execution.py`: 논문 Algorithm 1의 **고정 지연 시뮬레이션**. 추론 시작 시점
  관측으로 새 청크를 계산하고, 이전 청크로 d개 환경 step을 진행한 후 새 청크의
  d번째 인덱스부터 사용한다. `0 <= d <= s <= H-d`를 검증한다.

기본은 기존 D3IL 평가와 같은 **지연 0**이다. `--delay-steps`는 명시적으로
주입한 환경 시간상의 지연이며 실제 GPU 추론 시간과 다르다. 실시간 스레드,
가변 지연 추정과 하드웨어 제어기까지 구현한 것은 아니다. 지연이 있는 baseline은
동일 스케줄의 naive asynchronous 대조군이다. 첫 청크 생성은 두 방법 모두
환경 시간이 시작되기 전 warm-up으로 처리한다.

## VESSL에서 실행

기존 프로젝트 환경(Python 3.10/3.11, PyTorch 2.3.1, HardFlow 지정 commit,
D3IL)을 그대로 사용한다. 추가 패키지는 없다. 저장소 루트에서 실행한다.

```bash
source /root/d3il-work/venv/bin/activate
cd /root/d3il-work/yflow-d3il
git switch feature/method4
git pull --ff-only

# 수식·스케줄·지표 단위 테스트 (GPU/시뮬레이터 불필요)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s fm_rtc/tests -v

# 실제 두 EMA 모델의 baseline 일치·RTC 추론까지 검사
HARDFLOW_ROOT=/root/d3il-work/HardFlow PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s fm_rtc/tests -v

# Smoke: 양쪽 모두 같은 3개 context × 2회, 결과 폴더는 새 경로 사용
python -m fm_rtc.run_comparison --task pushing --n-contexts 3 --n-traj 2 \
  --device cuda:0 --output-root codex/rtc/pushing_smoke

# 본 평가: 30×4회 / 82×4회, 각각 baseline과 RTC 실행 후 비교
python -m fm_rtc.run_comparison --task pushing --device cuda:0 \
  --output-root codex/rtc/pushing_main
python -m fm_rtc.run_comparison --task inserting --device cuda:0 \
  --output-root codex/rtc/inserting_main

# 선택적 고정 지연 실험: 3 controller steps (기본 dt=0.035초이면 105ms)
python -m fm_rtc.run_comparison --task pushing --delay-steps 3 --device cuda:0 \
  --output-root codex/rtc/pushing_delay3
```

실제 venv/저장소 경로가 다르면 해당 부분만 바꾼다. 기본 경로 탐색이 안 되면
`--hardflow-root /.../HardFlow --d3il-root /.../d3il --data-dir /.../data/<task>_sincos_v1`
를 덧붙인다. `--run-dir`로 체크포인트 폴더를 지정할 수 있다.
`--dry-run`은 실행 명령만 출력한다. 출력 경로가 이미 있으면 덮어쓰지 않고 중단한다.

기본 설정은 horizon 16(체크포인트), 실행 8, ODE 10, beta 5, seed 0이다.
Pushing 최대 400, Inserting 최대 2000 환경 step. 동일 context·trajectory·seed마다
독립 난수 스트림을 사용하므로 이전 episode 길이가 다음 episode 노이즈를 바꾸지 않는다.
기존 90.8%/81.4% 기록과 난수 프로토콜이 달라 **이번 baseline을 새로 실행해 비교**한다.
Inserting은 기존 82개 검증 초기 배치이며 별도 held-out test 성능으로 부르지 않는다.

## 결과와 해석

각 방법의 폴더에 `results.json`, `protocol.json`, `episodes.jsonl`, `rollouts.npz`를
저장하고 상위 폴더에 `comparison.json`을 만든다. 체크포인트·정규화·context·코드
해시와 설정이 일치할 때만 비교한다. 중간 실패 시 완료된 episode 로그는 남지만
불완전한 결과로 비교하지 않는다.

- 기존 성공률, 모드 엔트로피, D3IL 점수, 목표 거리.
- 목표/실제 손 XY의 jerk RMS 및 청크 전환 경계 jerk RMS (m/s^3).
- 추종 오차, 경로 길이, 환경 시간, 평균 추론 시간. 추론 시간에는 RTC의 VJP도 포함.
- context 단위 paired bootstrap 95% CI. 한 context이면 CI를 계산하지 않는다.

jerk는 제어 주기(`scene.dt * n_substeps`)의 위치 3차 차분이다. 관절 jerk나
물리 substep의 고주파 진동은 아니다. 경계는 실제 새 청크가 활성화된 행동 인덱스로
기록한다. 계산할 수 없는 짧은 궤적 지표는 null이며, 비교에서 양쪽 유효 쌍만 쓴다.
성공률과 경로·시간을 함께 보아 정지로 인한 jerk 감소를 구분한다. 성공률 CI가 0을
포함한다고 성능 유지/비열등성이 증명된 것은 아니다.

이 구현은 D3IL의 행동 증분 공간에 RTC를 적용한 것이다. 재학습, 행동 clipping,
물리 projection, LiPo 후처리를 추가하지 않는다. 성능 수치는 VESSL 평가 후 확인한다.

## 로컬 검증

2026-11-04 Windows / Python 3.12 / PyTorch 2.6 CPU에서 수식, 시간 정렬,
고정 관측, gradient, SI jerk를 검증했다. 두 실제 baseline EMA 체크포인트에서
RTC off 및 첫 청크의 출력은 기존 `Policy.plan`과 원소별로 정확히 일치했다.
RTC guidance 활성화 시 출력이 유한하고 가중치가 바뀌지 않음도 확인했다.
실제 MuJoCo 성공률 평가는 로컬에서 실행하지 않았다.
