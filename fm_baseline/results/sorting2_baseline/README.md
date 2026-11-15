# Sorting-2 ID baseline

ID 성공률 **74% (296/400)**, context bootstrap 95% CI **68.5–79.25%**입니다. 30만 step에서 ID 조기종료하고, 고정 ID 개발셋으로 선택한 **20만 step EMA**를 보존했습니다. 학습 seed는 42입니다.

관측 12차원, 목표 손 XY 변화량 행동 2차원입니다. 학습 540개·검증 60개, train-only min/max 정규화를 사용했습니다. 원본 상태 미러 600개 중 공식 ZIP에서 직접 확보한 101개만 바이트 동일성을 대조했습니다. 공식 Sorting-2 test contexts의 앞 100개 ×4회로 최종 평가했습니다.

```bash
python fm_baseline/eval_sorting.py \
  --run-dir fm_baseline/results/sorting2_baseline \
  --ckpt baseline_ema.pt --weights ema \
  --d3il-root /path/to/d3il \
  --contexts-file fm_baseline/results/sorting2_baseline/id_contexts.pkl \
  --n-contexts 100 --n-traj 4 --max-steps 500 \
  --ode-steps 10 --execute-steps 8 --seed 0 --paired-noise \
  --output-root /tmp/sorting2-eval
```

`preprocess/prepare_sorting.py`로 원본 시연과 공식 train/eval 목록을 전처리한 후 `fm_baseline/train.py --task sorting2`로 학습할 수 있습니다. 학습 예산·EMA·모델 선택 규칙은 `protocol.json`, 학습 곡선 수치는 `metrics.csv`, 개발셋 평가 이력은 `early_stop_history.json`에 있습니다. 원본 400회 지표 감사 결과는 `id_audit.json`입니다.

두 모델 모두 TemporalUnet dim64, dim_mults(1,2,4,8), horizon16, execute8, Euler10, EMA .995입니다. `baseline_ema.pt`는 선택된 EMA tensor를 그대로 보존한 추론용 파일이며 `--weights ema`로 로드합니다. 원본 checkpoint와 공개 파일의 해시, tensor 동일성은 `provenance.json`에 기록했습니다. 추론은 저장소 내부 `fm_baseline.models`를 사용합니다. 공개 포장 과정에서 config의 실행 환경 경로를 이식 가능한 경로로 바꾸고 중복 온라인 가중치·optimizer를 제거했습니다. 원본 체크포인트 및 전체 rollout 아카이브는 로컬 실험 폴더에 보존했습니다.

현재 모델 로더로 두 체크포인트를 읽고 CPU에서 Euler10 계획 생성과 유한값을 확인했습니다. 위 평가 진입점은 저장소 내부 모델 API에 맞춰 이식했으며, 공개 포장 후 전체 시뮬레이션을 다시 실행하지는 않았습니다. 보고한 성공률은 보존된 원래 평가 결과입니다. 단일 학습 seed의 ID 시뮬레이터 결과입니다.

이번 원본 ID 평가의 context 파일은 이 폴더의 `id_contexts.pkl`이며 SHA256은 `d214f4c6777b4110bfdb40978396cec8c4d6b0bc0e59dd480a28ebc695390a48`입니다. 공용 `fm_baseline/contexts/sorting_2_test_contexts.pkl`은 별도로 생성한 다른 배치이므로 원래 평가 재현에 대신 사용하지 않습니다.
