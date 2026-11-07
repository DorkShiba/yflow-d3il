# Baseline LiPo 후처리와 smoothness 평가

기존 EMA 체크포인트를 고정하고 `none`(baseline)과 `lipo`(⑤)를 paired 비교한다. 추가 학습은 없다.

⑤ LiPo의 정의와 구현은 [`docs/METHODS.md`](../docs/METHODS.md) 2절과 `fm_constraint/lipo.py`에 있다. 이 문서는 **baseline 평가 스크립트에서 ⑤만 켜서** smoothness를 측정하는 도구를 설명한다. 5개 방법 조합 실험은 `fm_constraint`의 suite runner를 쓴다.

## 구현 범위

- 각 새 chunk에 고정 한도 minimum-jerk QP(`fm_constraint/lipo.jerk_qp`)를 적용한다. ① 제약은 넣지 않는다. 최근 실행한 목표 위치 최대 3개를 고정해 chunk 경계의 jerk도 목적함수에 포함한다.
- 각 시점·축의 위치 보정은 `--lipo-path-bound-m`(기본 0.3 mm) 이하다. Clarabel로 mm 단위에서 푼다.
- 이전 chunk와의 blend는 하지 않는다(2026-11-07 결정, codex `mode="qp"` 정의). 이전의 `blend` 모드와 `--lipo-blend-*` 옵션은 삭제했다.
- `none`은 원본 행동을 그대로 실행한다. QP 실패나 비유한값은 오류로 중단하며, 원본 행동으로 대체하지 않는다.
- 동기식·추론 지연 0·XY 위치 공간 적용이다. 논문의 관절 공간, 비동기 지연 스케줄링, spline 재현은 아니다.

## 실행

저장소 루트 기준이며, D3IL과 `requirements.txt` 환경이 필요하다.

```bash
python fm_baseline/run_lipo_comparison.py --task pushing --n-contexts 3 --n-traj 2 --device cuda:0
python fm_baseline/run_lipo_comparison.py --task inserting --dry-run      # 명령만 확인
```

- **runner:** 체크포인트 `baseline_ema.pt`, EMA, Euler 10스텝, 8스텝 실행, 같은 seed를 고정한다. 두 비교군을 순서대로 실행하고 `comparison_lipo.json`을 만든다. 출력은 기본 `fm_baseline/runs/lipo_comparison_<UTC>/`이며 Git에서 제외된다.
- **단독 실행:** `fm_baseline/eval_pushing.py ... --execute-steps 8 --paired-noise --postprocess lipo --lipo-path-bound-m 0.0003`. Pushing 평가 스크립트의 기본 실행 길이는 1이므로 `--execute-steps 8`을 명시한다.
- **병렬 실행:** 서버에서는 `run_lipo_sharded.py --output-root ... --d3il-root ... --n-contexts 30 --n-traj 4 --workers 9 --shard-contexts 3`으로 context를 나눠 돌린다. 합칠 때 context 범위의 중복·누락과 실행 조건 불일치를 거부한다.

## 지표

제어 간격은 `env.scene.dt * env.n_substeps`(0.035 s)에서 읽는다.

| 지표 | 정의 |
|---|---|
| desired/actual jerk RMS (m/s³) | 초기 위치를 포함한 위치 궤적의 `Δ³p / dt³` 크기 RMS |
| desired/actual seam jerk RMS | 재계획 경계를 가로지르는 4개 샘플 stencil의 jerk RMS |
| seam velocity jump RMS (m/s) | 경계에서 `‖ΔXY_new − ΔXY_old‖ / dt` |
| tracking gap, path length, duration | 정지해서 smoothness만 좋아지는 경우를 확인 |
| plan/postprocess/QP ms | 추론, 후처리, QP 시간을 따로 기록 |

- **파일별 내용:** `results.json`에는 집계 지표, 설정, 파일 SHA256(모델 코드 포함), D3IL commit이 들어간다. `smoothness.json`에는 episode별 지표와 plan별 QP 상태·보정량이 들어간다.
- **paired noise:** `--paired-noise`는 `(seed, context, trajectory)`별 난수 스트림을 쓴다.
- **비교:** `compare_smoothness.py`는 같은 checkpoint·설정·context·코드 hash의 결과만 받아 context 단위 paired bootstrap 95% CI를 계산한다.

## 검증

`python -m pytest tests/test_lipo.py tests/test_lipo_evaluation.py`

- 보정 한도, 목적함수 감소, 고정 과거 위치, baseline 원본 보존, solver 실패 처리
- SI jerk, 경계 stencil, 비교 조건 불일치 거부, shard 병합
- seed 14로 고정한 합성 chunk 40개에서 목표 jerk RMS가 78.7 → 41.4 m/s³로 줄었다. 합성 입력이며 실제 D3IL 결과가 아니다.

2026-11-04의 실제 D3IL 비교(Pushing 120회, Inserting 328회)는 이전 blend+QP 정의로 수행했다. 그 보고서는 git 이력(`experiments/lipo_*_20261104`, 커밋 `4ac4626` 이전)에 있다. **현재 정의로는 아직 다시 평가하지 않았다.**
