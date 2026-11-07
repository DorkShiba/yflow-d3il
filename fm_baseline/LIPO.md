# D3IL LiPo 구현과 smoothness 비교

기존 EMA 체크포인트를 고정하고 `none`, `blend`, `lipo`를 비교한다. 추가 학습은 없다.
구현 근거는 [LiPo 원 논문](https://arxiv.org/html/2506.05165v1)의 식 (4)–(6)이다.

## 구현 범위

- 원래 단위로 역정규화된 ΔXY를 현재 **목표 손 위치**에서 누적해 위치 청크를 만든다.
- 이전에 최적화한 청크의 미실행 위치를 절대 좌표로 보관한다. 16개 생성, 8개 실행이므로 다음 청크와 겹치는 구간은 8개다.
- 다음 청크의 앞 `blend_steps`개 위치를 이전 청크와 선형 혼합한다. 시간 0은 마지막 실행 목표 위치이고, 첫 미래 샘플 가중치는 `1/blend_steps`, 마지막은 1이다.
- Clarabel로 `min ||D3 [executed_history; reference + correction]||²`를 푼다. 최근 실제로 **실행한 목표 위치** 최대 3개는 고정해 경계 jerk도 목적함수에 포함한다. 시작 시 없는 과거 샘플은 만들어내지 않는다.
- correction에 샘플별 L∞ box bound를 적용한다. 내부 수치 계산만 mm로 스케일링하며 API·로그는 m 단위다. 고정 dt의 공통 목적함수 배율은 최적해에 영향이 없으므로 생략한다.
- 최적화 결과를 ΔXY로 환산해 기존 제어기에 전달한다. 나머지 위치 청크는 다음 재계획에 사용한다.
- `none`은 원본 행동 값을 그대로 실행한다. QP 실패, 비유한값, 한도 위반은 오류로 중단하며 성공한 LiPo처럼 집계하지 않는다.

이는 **동기식·추론 지연 0·이산 XY 위치 공간**의 LiPo 적용이다. 논문의 관절각 공간, 비동기 지연 스케줄링, 400 Hz quintic spline 전체 재현은 아니다. D3IL의 기존 제어 주기와 저수준 제어기는 유지한다. 장애물·동역학 안전 제약은 넣지 않았다.

기본값 `blend_steps=4`, `blend_bound_m=0.002`, `path_bound_m=0.0003`은 검증 전 시작값이다. 논문의 rad 값을 m로 변환한 수치가 아니다. 첫 청크는 이전 청크가 없으므로 blending 없이 모든 위치에 path bound를 적용한다. bound는 **blending한 reference 대비 QP 보정량**이며, raw 정책 대비 총 변위 한도가 아니다. 두 변위를 각각 기록한다.

## 실행

기존 README의 D3IL·학습 체크포인트 실행 환경에서, 저장소 루트 기준:

```bash
pip install -r requirements-lipo.txt
python fm_baseline/run_lipo_comparison.py --task pushing --n-contexts 3 --n-traj 2 --device cuda:0
```

먼저 3×2 smoke 평가를 하고, 조건을 확정한 뒤 같은 명령의 `--n-contexts 30 --n-traj 4`로 확대한다. Inserting은 `--task inserting`을 사용한다. `--dry-run`은 실행 예정 명령만 출력한다. 별도 경로는 `--d3il-root`, `--run-dir`, `--output-root`로 지정한다. 출력 루트는 기존 폴더를 덮어쓰지 않는다.

실행기는 체크포인트 `baseline_ema.pt`, EMA, Euler 10스텝, 8스텝 실행, 같은 seed를 고정한다. 세 비교군을 순서대로 실행하고 `comparison_blend.json`, `comparison_lipo.json`을 만든다. 출력은 기본 `fm_baseline/runs/lipo_comparison_<UTC>/`이며 Git에서 제외된다.

한 비교군만 실행하려면:

```bash
python fm_baseline/eval_pushing.py \
  --run-dir fm_baseline/results/pushing_baseline --ckpt baseline_ema.pt \
  --execute-steps 8 --ode-steps 10 --n-contexts 30 --n-traj 4 \
  --seed 0 --paired-noise --postprocess lipo \
  --lipo-blend-steps 4 --lipo-blend-bound-m 0.002 --lipo-path-bound-m 0.0003
```

`--postprocess none`이 베이스라인, `blend`는 QP 없는 혼합이다. Pushing 평가 스크립트의 기존 기본 실행 길이는 1이므로 단독 실행 시에도 `--execute-steps 8`을 명시한다.

## 지표와 결과 해석

제어 시간 간격은 하드코딩하지 않고 `env.scene.dt * env.n_substeps`에서 읽어 기록한다. 현재 공식 설정은 0.001×35 = 0.035초다.

| 지표 | 정의 |
|---|---|
| desired/actual jerk RMS (m/s³) | 초기 위치 포함 위치 궤적의 `Δ³p / dt³` 벡터 크기 RMS |
| desired/actual seam jerk RMS | 재계획 경계를 가로지르는 4개 위치 샘플 stencil의 jerk RMS |
| seam velocity jump RMS (m/s) | 경계에서 `||ΔXY_new - ΔXY_old|| / dt`의 RMS |
| action delta RMS (m) | 행동 차분. 물리 jerk와 구분 |
| acceleration/velocity RMS | 위치 2차/1차 차분에 올바른 시간 단위 적용 |
| tracking gap, path length, duration | 추종 오차와 이동량·소요 시간. 단순 정지로 smoothness만 좋아지는 경우 확인 |
| success rate | 기존 태스크 판정 유지 |
| plan/postprocess/QP ms | 정책 추론, 후처리, QP 구성·풀이 시간을 별도 기록 |

실제 손 위치 지표도 **행동 주기로 샘플링된 XY**의 유한차분이다. 물리 substep의 고주파 진동이나 관절 jerk 측정은 아니다. 별도 필터는 적용하지 않는다. 위치 로그는 3차 차분 정밀도를 위해 float64로 저장한다. 짧은 rollout의 계산 불가능 지표는 0 대신 null로 남긴다.

`results.json`에는 집계 지표·실행 설정·파일 SHA256·외부 저장소 commit을, `smoothness.json`에는 episode별 지표와 plan별 QP 상태·보정량·시간을 저장한다. `rollouts.npz`에는 기존 궤적과 함께 초기 위치 및 재계획 마스크를 저장한다. 첫 호출 시간에는 모듈 import 등 초기 비용이 들어갈 수 있다. 합계 실행 비용은 `plan_ms + postprocess_ms`이며 `qp_ms`는 후처리에 포함된다.

`--paired-noise`는 `(seed, context_index, trajectory_index)`에서 episode별 난수 스트림을 만들어, 앞 episode 길이가 다음 episode의 정책 노이즈를 바꾸지 않게 한다. 같은 episode에서는 재계획 번호별로 동일 노이즈를 사용한다. 상태와 성공 여부는 후처리에 따라 달라질 수 있다.

비교 도구는 체크포인트·정규화·설정·context 내용·코드 해시·실행 조건이 같고 paired noise를 사용한 결과만 받는다. episode 평균을 context별로 묶어 paired bootstrap 95% CI를 계산한다. 한 context뿐이면 CI를 보고하지 않는다. 지표가 없는 episode는 양쪽 모두에서 제외하고 유효 개수를 표시한다.

```bash
python fm_baseline/compare_smoothness.py \
  --baseline /absolute/path/to/baseline_eval \
  --candidate /absolute/path/to/lipo_eval \
  --output /absolute/path/to/comparison.json
```

기존 `fm_pc/compute_metrics.py`의 `jerk` 필드는 `mean ||Δaction|| / a_max`인 과거 proxy다. LiPo 평가는 새 SI 단위 지표를 사용한다. 기존 성공률 90.8% 기록은 참고값이며, 새 paired 프로토콜의 baseline은 반드시 다시 실행한다. 전체 episode 지표는 성공 여부에 따른 길이 차이의 영향을 받으므로 성공률·시간·경계 지표를 함께 해석한다.

## 검증 상태

```bash
python -m unittest discover -s tests -v
```

2026-11-04 로컬 검증: Python 3.14, NumPy 2.5.4, SciPy 1.18.1, Clarabel 0.11.1. 이는 NumPy 1.26.4 기반 D3IL 배포 환경을 대체하지 않는 수치 테스트 환경이다.

- QP 보정 한도, 목적함수 감소, 고정된 과거 위치, 절대 좌표 overlap 정렬, baseline 원본 행동 보존, solver 실패 처리를 검증했다.
- cubic 위치의 정확한 SI jerk, 경계 stencil, 짧은 episode 처리, 비교 조건 불일치 거부를 검증했다.
- 가짜 정책·가짜 환경으로 두 **실제 평가 루프**의 세 비교군, 로그 저장, paired 비교 연결을 검증했다. 이 테스트는 MuJoCo 실행을 검증하지 않는다.
- seed 14로 고정한 합성 noisy chunk 40개(16개 생성·8개 실행)의 목표 jerk RMS는 none 78.666, blend 72.138, lipo 21.670 m/s³였다. 경계 속도 변화 RMS는 각각 0.05488, 0.04637, 0.01551 m/s였다. 이는 합성 입력의 수치 확인이며 학습된 정책·실제 D3IL의 개선 결과가 아니다.
- 이후 2026-11-04 RTX 3090 서버에서 Python 3.10.12 / NumPy 1.26.4 / PyTorch 2.3.1+cu121 환경의 테스트 16개 및 실제 Pushing 평가 120회×3군을 완료했다. 실제 손 jerk RMS 25.3% 감소, 성공률 106/120 → 115/120. [실험 보고서](../experiments/lipo_pushing_20261104/REPORT_KO.md)와 원본 근거를 참고한다. 이후 Inserting 328회×3군도 완료했다. 실제 손 jerk RMS는 17.3% 감소했으나 성공률은 263/328 → 256/328이며, 차이의 95% CI가 0을 포함해 성공률 유지·개선은 입증되지 않았다. [Inserting 보고서](../experiments/lipo_inserting_20261104/REPORT_KO.md)를 참고한다. OOD·추론 지연 평가는 아직 수행하지 않았다.

## Context 분할 실행

GPU 메모리와 CPU 여유가 있는 서버에서는 `run_lipo_sharded.py`로 같은 평가를 병렬화할 수 있다.

```bash
python fm_baseline/run_lipo_sharded.py \
  --output-root /absolute/path/to/new-run \
  --d3il-root /absolute/path/to/d3il \
  --n-contexts 30 --n-traj 4 --workers 9 --shard-contexts 3
```

Pushing의 `--context-start`는 공식 context 배열의 시작 인덱스다. paired RNG는 local 인덱스에 시작 인덱스를 더하므로 분할 방식과 무관하게 같은 난수를 사용한다. 합칠 때 context 범위 중복·누락과 실행 조건 불일치를 거부한다. 원본 shard 결과는 보존하며 `combined/`에 context 인덱스를 복원한 궤적과 지표를 저장한다. 분할 RNG 일치, 로그의 정확한 재결합도 테스트한다.
