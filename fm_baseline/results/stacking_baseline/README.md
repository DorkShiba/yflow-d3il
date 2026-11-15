# Stacking ID baseline

ID 성공률 **34.5% (138/400)**, context bootstrap 95% CI **29.75–39.25%**입니다. 50만 step에서 ID 조기종료하고, 고정 ID 개발셋으로 선택한 **35만 step EMA**를 보존했습니다. 학습 seed는 42입니다.

관측 23차원, 관절 목표 변화량 7개와 다음 기록의 그리퍼 폭을 합친 행동 8차원입니다. 원본 vision-state 미러 1,094개를 release 제공 학습 1,001개·검증 93개로 분할했고 train-only min/max 정규화를 사용했습니다. 모든 원본이 미러 release 해시와 일치합니다. 공식 ZIP 직접 대조는 할당량 제한으로 0개입니다.

평가 context는 공식 manager 분포에서 새로 생성한 142개로 test [0,100), dev [100,140), smoke [140,142)를 고정했습니다. 공식 배포 test 파일과 다른 새 ID 표본을 평가했습니다. 전체 성공은 세 블록이 목표 XY 반경 6cm 이내이고 모든 쌍의 Z 차이가 3cm 초과인 순간 판정입니다. success_1/2는 1/2개 블록의 목표 XY 영역 진입 이력입니다.

```bash
python fm_baseline/eval_stacking.py \
  --run-dir fm_baseline/results/stacking_baseline \
  --ckpt baseline_ema.pt --d3il-root /path/to/d3il \
  --contexts-file fm_baseline/results/stacking_baseline/id_contexts.pkl \
  --n-contexts 100 --n-traj 4 --max-steps 1000 \
  --ode-steps 10 --execute-steps 8 --seed 0 \
  --output /tmp/stacking-eval
```

`preprocess/prepare_stacking.py`로 원본 시연과 release metadata를 전처리한 후 `fm_baseline/train.py --task stacking`으로 학습할 수 있습니다. 학습 예산·EMA·모델 선택 규칙은 `protocol.json`, 학습 로그는 `metrics.csv`, 개발셋 평가 이력은 `early_stop_history.json`, 원본 400회 지표 감사는 `id_audit.json`에 있습니다. `learning_curve.png`에 학습곡선과 최종 결과를 담았습니다. 기존 평가의 계획 시간은 GPU 4개 worker 혼잡을 포함합니다.

두 모델 모두 TemporalUnet dim64, dim_mults(1,2,4,8), horizon16, execute8, Euler10, EMA .995입니다. `baseline_ema.pt`는 선택된 EMA tensor를 그대로 보존한 추론용 파일이며 `--weights ema`로 로드합니다. 원본 checkpoint와 공개 파일의 해시, tensor 동일성은 `provenance.json`에 기록했습니다. 추론은 저장소 내부 `fm_baseline.models`를 사용합니다. 공개 포장 과정에서 config의 실행 환경 경로를 이식 가능한 경로로 바꾸고 중복 온라인 가중치·optimizer를 제거했습니다. 원본 체크포인트 및 전체 rollout 아카이브는 로컬 실험 폴더에 보존했습니다.

현재 모델 로더로 두 체크포인트를 읽고 CPU에서 Euler10 계획 생성과 유한값을 확인했습니다. 위 평가 진입점은 저장소 내부 모델 API에 맞춰 이식했으며, 공개 포장 후 전체 시뮬레이션을 다시 실행하지는 않았습니다. 보고한 성공률은 보존된 원래 평가 결과입니다. 단일 학습 seed의 ID 시뮬레이터 결과입니다.

## 관절 PF 캐시

[joint_pf_cache.npz](joint_pf_cache.npz)는 2026-11-15 ID ①②③⑤ 비교에 사용한 원본 frozen 캐시입니다. 약313 KiB이며 train 시연에서 선택한 3,000개 7관절 support, 2,951개 고밀도 anchor와 정규화 정보를 담았습니다. 검증 split은 캐시 구성에 사용하지 않았습니다. 공개 normalizer와 배열·해시 대조를 마쳤으며 [provenance](joint_pf_cache.provenance.json)에 기록했습니다.

[조합 평가 실행 방법](../../../docs/results/sorting2_stacking_combo29_id_20261115/README.md#stacking-재실행)의 `--cache fm_baseline/results/stacking_baseline/joint_pf_cache.npz`로 바로 사용할 수 있습니다.

Stacking을 새로 clone하여 baseline vs ①②③④⑤를 실행하려면 [팀원 실행 안내](../../../docs/stacking_team_run.md)를 참고합니다. 모델·캐시 파일 목록, 설치, smoke와 전체 paired 비교 명령을 제공합니다. 기존 ①②③⑤ 보고 결과와 ④를 추가한 실행은 구분합니다.
