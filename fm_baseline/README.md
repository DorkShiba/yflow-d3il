# 순수 플로우 매칭 베이스라인 (Pushing · Inserting)

작성일: 2026-10-16

물리 제약 보정이 없는 순수 플로우 매칭 모델을 D3IL Pushing·Inserting 상태 데이터로 학습하고, D3IL 시뮬레이터에서 평가한 기록이다. 이 체크포인트들이 32개 제약 조합 실험의 공통 기준 모델이 된다. 아래 `/root/...` 경로는 **VESSL 워크스페이스 경로**다. 두 베이스라인의 설정·정규화 값·학습 로그·평가 결과는 `results/`에 있다.

## 1. 결과 요약

| | Pushing | Inserting |
|---|---|---|
| 모델 | Temporal U-Net, dim 32, dim_mults (1, 4, 8), 약 368만 파라미터 | Temporal U-Net, dim 64, dim_mults (1, 2, 4, 8), 약 1,579만 파라미터 |
| 실행 폴더 | `runs/pushing_main_nosn_20261014T010928Z` | `runs/inserting_abl_d64_nosn_20261014T105353Z` |
| 최종 검증 손실 (EMA) | 0.0940 | 0.1007 |
| 평가 context | 공식 `test_contexts.pkl` 앞 30개 × 4회 | 검증 분할 시연 82개의 초기 배치 × 4회 |
| 성공 기준 / 최대 스텝 | 블록 2개가 목표 5cm 이내 / 400 | 블록 3개가 각 목표 1cm 이내 / 2,000 |
| **성공률** | **90.8%** (109/120) | **81.4%** (267/328) |
| 엔트로피 / 점수 | 0.491 / 0.700 (모드 4개) | 0.351 / 0.583 (모드 6개) |
| 평균 거리 (블록–목표) | 0.044m | 0.033m |
| 평균 스텝 | 237.5 | 1,345 (성공 시 1,205, 원본 시연 평균 982) |
| 모드 분포 (성공) | 0: 37, 1: 45, 2: 7, 3: 19 | 1: 34, 2: 68, 3: 44, 4: 50, 5: 36, 6: 35 |
| 부분 진척 | – | 블록 1개 97.6%, 2개 91.8%, 3개 82.3% |

점수는 D3IL 공식 정의인 `0.5 × (성공률 + 엔트로피)`이고, 엔트로피는 같은 context 안 성공 rollout의 모드 분포를 모드 수로 정규화한 값의 평균이다. 두 태스크는 블록 수, 허용 오차, 장애물 유무가 달라서 성공률을 서로 직접 비교하지 않는다. 제약 실험은 **같은 태스크 안에서 같은 체크포인트와 같은 평가 조건**으로 비교한다.

## 2. 공통 설정

- 백본: HardFlow 공식 저장소 `d3il` 브랜치, commit `a85105080dd0a28dca76b49ac0414f275de146ed`의 `TemporalUnet`과 `FlowMatcher`(CFM)를 수정 없이 사용.
- 입력: `[actions, obs]` 순서의 horizon 16 윈도우. 시연 경계를 넘는 윈도우와 패딩은 사용하지 않음.
- 조건: t=0 관측을 `apply_conditioning`으로 고정. 이는 조건부 생성이며 물리 제약이 아니다.
- 정규화: 학습 분할만으로 차원별 min-max를 계산해 [-1, 1]로 변환. 범위가 1e-8보다 작은 채널은 스케일 1.
- 학습: 100만 스텝, 배치 32, Adam lr 2e-4, 전체 스텝에 걸친 단일 Cosine, EMA 0.995, weight decay 없음, seed 42.
- 공식 HardFlow 학습 설정은 Cosine의 T_max가 10만이라 학습 중 lr이 여러 번 내려갔다 올라간다. 여기서는 의도를 명확히 하려고 전체 스텝에 걸친 단일 Cosine으로 바꿨다.
- Spectral Normalization: **사용하지 않음.** 처음에는 모델 기본값으로 넣기로 했다가 팀 결정으로 제외했다. 참고로 HardFlow U-Net은 conv 뒤에 GroupNorm이 붙어 있어, SN을 걸어도 그 층에서는 효과가 대부분 상쇄된다. SN을 켠 실행은 35만 스텝에서 중단했고 `runs/*_main_sn_*`에 남아 있다.
- 생성: 공식 `ConditionedODESolver`(Euler, `value=None`)로 10스텝. 물리 제약 보정 없음.

## 3. 평가 프로토콜

- 매 계획 시점에 장면에서 최신 상태를 직접 읽어 학습과 같은 관측(목표 손 XY, 실제 손 XY, 블록별 X, Y, sin(yaw), cos(yaw))을 만든다. `env.step()`이 반환하는 관측은 물리 계산 전 값이라 쓰지 않는다.
- 16스텝 궤적을 생성하고 **앞 8스텝의 행동을 연달아 실행**한 뒤 다시 생성한다. 행동은 목표 손 XY 변화량이며, 이전 목표 위치에 누적해 `[x, y, 고정 z, 0, 1, 0, 0]`으로 환경에 넘긴다. 첫 목표 위치는 리셋 직후의 실제 손 위치다. 행동 클리핑 등 보정은 하지 않는다.
- 실행 스텝 수 비교 (Pushing, 테스트 context 30개 × 1회): 매 스텝 재생성은 짧은 평가 5개 중 2개 성공, 4스텝 실행 93.3%, **8스텝 실행 96.7%**. 매 스텝 재생성은 스텝마다 다른 모드를 골라 망설이거나, 목표 위치가 실제 손에서 멀어져 폭주하는 실패가 많았다. 실행 방식은 제약이 아니라 정책 실행 설정이므로 모든 제약 조합에 동일하게 적용한다.
- 관측 형식 검증: 검증 시연 5개의 초기 상태를 환경에 세팅해 만든 관측이 전처리 데이터 `obs[0]`과 일치함을 확인했다. 블록 위치·각도와 목표 손 위치는 오차 0, 실제 손 XY는 최대 0.06mm(검증 중 1ms 물리 진행에 따른 차이).

## 4. Inserting 모델 크기를 키운 이유

같은 설정(dim 32, (1, 4, 8))으로 100만 스텝 학습한 Inserting 모델은 검증 손실이 0.193에서 멈췄고, 생성 결과가 기준선보다 나빴다.

- 생성 행동의 크기가 실제의 약 2배였고 32%가 데이터 최댓값(0.00833)을 넘었다. 0 행동 기준선보다 RMSE가 컸다.
- 생성 궤적의 마지막 프레임에서 목표 손 위치가 평균 0.21m 튀었다.
- 학습 손실과 검증 손실이 같아 과적합이 아니라 학습 부족이었고, 10만 스텝 이후 거의 개선되지 않았다.
- 데이터는 정상이었다. 행동은 목표 손 위치 변화량과 6e-8 이내로 일치했고, 행동이 바뀌는 비율도 Pushing과 비슷했다.

10만 스텝 비교 실험 결과는 다음과 같다.

| 설정 | 검증 손실 (EMA) | 행동 RMSE | 생성 행동 크기 (실제 0.0034) |
|---|---|---|---|
| 기존 dim 32 | 0.213 | 0.0094 | 0.0071 |
| attention 추가 | 0.206 | 0.0094 | 0.0072 |
| **dim 64, (1, 2, 4, 8)** | **0.116** | **0.0054** | **0.0034** |

dim 64 실행을 그대로 100만 스텝까지 이어서 학습해 Inserting 베이스라인으로 확정했다. 같은 설정으로 Pushing도 학습했지만 검증 손실이 0.124로 dim 32(0.094)보다 나빴고 학습 손실 0.076과 차이가 커서 과적합으로 판단했다. 그래서 **Pushing은 dim 32, Inserting은 dim 64**를 사용한다. 채널 폭과 다운샘플 단계를 함께 바꿨으므로 둘 중 어느 쪽이 결정적이었는지는 분리하지 않았다.

## 5. 실패 분석

### Pushing (실패 11/120, 모두 400스텝 시간 초과)

- 아깝게 실패 7건: 두 블록을 목표 근처(평균 거리 4~10cm)까지 밀었지만 5cm 안에 넣지 못함.
- 목표 위치 폭주 3건: 목표 손 위치가 실제 손에서 24~30cm까지 벌어지고 손이 거의 멈춤.
- 한 블록에서 맴돎 1건.

### Inserting (실패 61/328)

- 시간 초과 58건, 일찍 끝났는데 실패 3건. 후자는 환경이 물리 계산 직전에 종료를 판정하고 직후 블록 하나가 1cm 밖으로 밀려난 경우로, 공식 환경 코드의 동작이다.
- 넣은 블록 수: 0개 8건, 1개 19건, 2개 31건, 3개(동시 아님) 3건.
- 거의 다 넣음(평균 거리 2cm 미만) 16건.
- 손이 멈춤(마지막 300스텝 이동 5cm 미만) 34건, 목표–손 거리 10cm 초과 30건. 미로 벽에 손이 막힌 동안 목표 위치가 계속 벽 쪽으로 이동하는 패턴으로 추정한다. 벽 좌표와 궤적을 겹쳐 확인하는 작업은 남아 있다.

목표 위치 폭주와 벽 근처 정체는 작업 공간·장애물 여유·추종 거리 제약이, 아깝게 실패한 경우는 종단 보정 계열 제약이 겨냥하는 유형이다.

## 6. 파일

| 파일 | 역할 |
|---|---|
| `train.py` | 학습. `--sn`, `--dim`, `--dim-mults`, `--attention`, `--scheduler cosine`, `--ema-decay`, `--save-every`, `--time-limit-hours`, `--resume` 지원 |
| `sample.py` | 검증 윈도우에 대한 오프라인 생성 검사. 형태, 유한값, t=0 조건 유지, 행동·관측 RMSE와 기준선 출력 |
| `eval_pushing.py` | Pushing 시뮬레이터 평가. `--check-obs`로 관측 형식 검증 |
| `eval_inserting.py` | Inserting 시뮬레이터 평가. 검증 분할 초기 배치를 context로 사용 |
| `results/` | 두 베이스라인의 학습된 가중치 `baseline_ema.pt`, `config.json`, `normalizer.npz`, `metrics.csv`, 평가 결과 JSON |
| `../preprocess/` | `data/`를 만든 전처리·검증 스크립트 |

학습 실행 폴더에는 `config.json`, `normalizer.npz`, `metrics.csv`, `last.pt`(재개용 전체 상태), `ckpt_XXXXXXX.pt`(2.5만 스텝마다 모델과 EMA 가중치)가 저장된다. 평가 폴더 `eval_*`에는 `results.json`과 rollout 궤적 `rollouts.npz`가 저장된다. 재개용 전체 체크포인트와 rollout 궤적은 용량이 커서 저장소에 올리지 않는다. 베이스라인 EMA 가중치는 `results/<태스크>_baseline/baseline_ema.pt`에 있으며, 원본 `last.pt`(100만 스텝)에서 EMA 가중치만 꺼내 저장한 것이다.

`train.py`의 `--resume`은 옵티마이저, EMA, 난수 상태까지 복원한다. 중간에 끊고 재개한 결과가 끊지 않은 학습과 가중치까지 일치함을 확인했다.

## 7. 실행 방법

```bash
source /root/d3il-work/codex/venv/bin/activate
cd /root/d3il-work/codex/fm_baseline
COMMON="--steps 1000000 --batch-size 32 --lr 2e-4 --scheduler cosine --ema-decay 0.995 --log-every 5000 --save-every 25000 --tag main --time-limit-hours 11"
python train.py --task pushing --device cuda:0 $COMMON
python train.py --task inserting --device cuda:1 $COMMON --dim 64 --dim-mults 1,2,4,8
```

```bash
python sample.py --run-dir runs/<실행 폴더>
python eval_pushing.py --run-dir runs/<Pushing 실행 폴더> --check-obs 5
python eval_pushing.py --run-dir runs/<Pushing 실행 폴더> --n-contexts 30 --n-traj 4 --execute-steps 8 --tag exec8_main
python eval_inserting.py --run-dir runs/<Inserting 실행 폴더> --check-obs 5
python eval_inserting.py --run-dir runs/<Inserting 실행 폴더> --n-traj 2 --seed 0 --tag exec8_main_s0
python eval_inserting.py --run-dir runs/<Inserting 실행 폴더> --n-traj 2 --seed 1 --tag exec8_main_s1
```

Inserting 본 평가는 시드 0과 1로 각각 82 × 2회를 동시에 실행한 뒤 합쳐 82 × 4회로 계산했다. 각 rollout은 약 73초, 전체 약 3.5시간이 걸렸다. 학습 속도는 RTX 3090 기준 dim 32 약 33 it/s(약 8.3시간), dim 64 약 25~27 it/s(약 10시간)였다.

## 8. 환경 참고

- VESSL venv: Python 3.10.12, PyTorch 2.3.1+cu121, NumPy 1.26.4, Gym 0.23.1, MuJoCo 2.3.2, einops 0.8.1. HardFlow의 `requirements.txt`는 통째로 설치하지 않았다.
- 워크스페이스를 재시작하면 시스템 패키지가 사라진다. 시뮬레이터 평가 전에 `apt-get update && apt-get install -y libgl1 libglib2.0-0`이 필요할 수 있다. tmux도 같은 방식으로 다시 설치한다.
- D3IL 공식 Inserting 환경 파일은 `d3il_sim.*` 경로로 import하고 시뮬레이터 내부는 `environments.d3il.d3il_sim.*`를 사용해, 수정하지 않은 저장소에서는 모듈이 두 벌 로드되어 `'Box' object has no attribute 'mj_load'` 오류가 난다. `eval_inserting.py`는 D3IL 코드를 바꾸지 않고 두 경로를 같은 모듈로 연결한다.
- Inserting 평가 초기 배치는 `contexts/inserting_val.pkl`(검증 분할 82개 시연의 `context`, 첫 관절 자세, 목표 손 궤적)에서 읽는다. 이 파일은 `eval_inserting.py --export-contexts contexts/inserting_val.pkl`로 원본 pkl에서 만들었다. 파일에 없는 시연은 `--raw-root`(기본 `/root/d3il-work/raw/inserting`) 아래 원본을 재귀 검색한다.
- `config.json`의 경로가 없는 환경에서는 저장소 기준 위치(`<저장소>/data`, `<저장소>/../HardFlow`, `<저장소>/../d3il`)를 쓴다. 학습 결과는 기본으로 `fm_baseline/runs/`에 저장된다.

## 9. 남은 작업

1. 태스크별 실제 물리 제약 정의: 작업 공간 경계, 장애물·벽 여유 거리, 스텝당 최대 이동량, 목표–손 추종 거리 등.
2. 5개 제약 주입 방식 확정. 추론 시 적용하는 방식과 학습 시 적용하는 방식이 섞여 있는지 확인.
3. 제약 위반 지표를 rollout 궤적(`rollouts.npz`)에서 계산하는 스크립트 작성.
4. Inserting 실패 rollout과 미로 벽 좌표를 겹친 시각화.
