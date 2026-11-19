# Stacking baseline vs ①②③④⑤ 팀원 실행 안내

45만 step 절대좌표·상대좌표 모델로 같은 32조합을 실행하려면
[관측 좌표 ablation 실행 가이드](stacking/coordinate_ablation/README.md)를 참고한다.

기존 학습 모델을 고정한 **baseline(모두 off) vs 다섯 방법을 모두 켠 조합(31)** 비교입니다. 방법 하나씩의 독립 ablation이나 32조합 전체 실험은 이 명령에 포함되지 않습니다. 기존에 보고한 성공률·jerk 비교는 **①②③⑤ 조합(29)** 결과이며, ④를 추가한 조합(31)의 성능 결과로 사용할 수 없습니다.

## 필요한 파일

아래 파일은 모두 Git에 포함되어 있어 일반 `git clone`으로 받습니다. 별도 모델 다운로드, Git LFS, 원본 학습 시연 다운로드나 재학습은 필요하지 않습니다.

| 경로 | 역할 |
|---|---|
| `fm_baseline/results/stacking_baseline/baseline_ema.pt` | 35만 step에서 선택한 기존 EMA 모델, 약60 MiB |
| 같은 폴더의 `config.json`, `normalizer.npz` | 관측23·행동8·horizon16 구조 및 train-only 정규화 |
| 같은 폴더의 `id_contexts.pkl` | 고정142 ID contexts; test는 앞100개, dev는100–139, smoke는140–141 |
| 같은 폴더의 `joint_pf_cache.npz` | 실제 조합29 실험에 사용한 train-only 관절 PF 캐시, 약313 KiB |
| 같은 폴더의 `sha256.json`, `provenance.json`, `joint_pf_cache.provenance.json` | 파일 무결성과 출처 |
| `configs/stacking_combo31.json` | ①②③④⑤ 설정, ④ beta=5.0 |
| `configs/stacking_combo29.json` | 기존 ①②③⑤ 재현 설정 |
| `fm_constraint/joint.py`, `eval_stacking.py` 및 저장소 내부 모듈 | 관절 공간 방법과 paired 평가 |

**외부에 필요한 것은 D3IL 시뮬레이터 checkout, Python 의존성 및 OS 라이브러리입니다.** 학습 데이터 경로가 `config.json`에 남아 있어도 이 추론에는 사용하지 않습니다. ④에는 별도 가중치·캐시가 필요하지 않습니다.

## 1. Linux 실행 환경 준비

Python3.11과 NVIDIA GPU가 있는 Linux 환경을 기준으로 합니다. CUDA 드라이버는 별도 설치되어 있어야 합니다. 의존성 파일은 실행에 사용한 직접 의존성을 고정하며 전체 OS/CUDA 환경을 대체하지 않습니다.

```bash
mkdir -p ~/yflow-work
cd ~/yflow-work
git clone https://github.com/DorkShiba/yflow-d3il.git
git clone https://github.com/ALRhub/d3il.git
git -C d3il checkout 1d9c71850d4fc1477cee17b557c8b97d70b13071
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r yflow-d3il/requirements.txt
cd yflow-d3il
```

관리자 권한이 있는 Ubuntu에서 필요한 시스템 라이브러리가 없으면 `sudo apt-get update` 후 `sudo apt-get install -y libgl1 libglib2.0-0`을 실행합니다. D3IL은 `pip install`하지 않고 checkout 경로로 사용합니다. D3IL checkout에 임시 XML을 쓸 수 있어야 합니다. 이 안내에 계정·서버 접속 정보는 포함하지 않습니다.

## 2. 파일·모델 사전 검사

```bash
python -m fm_constraint.check_stacking_bundle --plan-check --device cpu
```

모델·정규화·context·PF 캐시 해시, 관측/행동 규격, 실제 모델의 계획 생성, 관절 QP와 ④ VJP 호출을 검사합니다. 이 검사는 시뮬레이터 rollout이나 성능 평가가 아닙니다.

## 3. 짧은 시뮬레이터 smoke

```bash
bash scripts/run_stacking_comparison.sh \
  --d3il-root ../d3il --device cuda:0 \
  --start 140 --count 2 --repeats 1 --steps 24 \
  --output fm_constraint/runs/stacking_combo31_smoke
```

baseline과 `joint_combo31` 각각2회가 실행됩니다. `COMPLETE.json`은 파일 저장 완료 표시입니다. 각 arm의 `results.json`에서 `failure`가 없는지 확인합니다. 조합 arm의 `rtc_guided_plans`와 `raw_reports.json.gz`의 `rtc_guided`, `rtc_calls`로 실제 ④ 사용 여부를 확인합니다. 필터가 매번 행동을 수정하면 참조가 폐기되어 ④가 적용되지 않을 수 있으므로, 활성화 설정과 실제 호출을 구분합니다. 24-step 결과를 성공률 추정으로 쓰지 않습니다.

## 4. ID 전체 paired 비교와 요약

```bash
bash scripts/run_stacking_comparison.sh \
  --d3il-root ../d3il --device cuda:0 \
  --start 0 --count 100 --repeats 4 --steps 1000 \
  --output fm_constraint/runs/stacking_combo31_id

python -m fm_constraint.summarize_stacking \
  --input fm_constraint/runs/stacking_combo31_id
```

100 contexts×4 repeats×2 arms=800 rollouts입니다. 동일 context/repeat의 난수는 CPU hash seed로 짝지으며, 기존 EMA와 normalizer를 그대로 사용합니다. horizon16, execute8, Euler10, dt0.03s, 모사 추론 지연0입니다. 순차 실행을 권장합니다. GPU를 공유하면 compute time에 혼잡이 포함됩니다.

각 arm의 `results.json`, 원본 `rollouts.npz`, `raw_reports.json.gz`와 전체 `summary.json`, `rollout_metrics.json`이 남습니다. 요약은 성공률, 실제 손 XYZ jerk RMS(m/s³), 청크 경계 관절 명령 속도 변화 RMS(rad/s), 계획 시간(ms), rollout별 필터 총시간(ms), 실패 수와 context-paired bootstrap95% CI를 포함합니다. 짧은/실패 rollout의 jerk·경계 지표는 없을 수 있어 유효 pair 수도 기록합니다. 실행 시간 전체나 필터 시간을 계획 시간에 포함시킨 것으로 해석하지 않습니다. Safety 지표는 비교 대상에서 제외합니다.

기존 조합29를 비교하려면 위 명령에 `--settings configs/stacking_combo29.json`을 추가하고 **다른 output**을 사용합니다. wrapper의 기본 설정을 마지막 인자가 덮어씁니다. 출력 서명은 모델·정규화·context·캐시·설정·평가 소스가 다르면 재사용을 거부합니다. 일치하는 완료 arm은 건너뛰고, 중간에 끊긴 arm은 `.partial.*`로 보존한 뒤 그 arm 전체를 다시 평가합니다. 같은 output을 여러 프로세스에서 동시에 실행하지 않습니다.

## 관절 공간 ④와 해석 범위

①관절 위치·가속도 공동 QP, ②train 관절 밀도 guidance, ③실행 관절 필터, ⑤관절 명령 jerk QP는 기존 조합29와 같습니다. ④는 메인의 RTC soft-mask와 VJP 식을 7관절 증분 참조에 적용합니다. 직전16-step 계획 중 실행한8개 이후의 미실행8개를 참조하며, 첫 계획·새 episode·③의 수정 직후에는 참조가 없습니다. 그리퍼 행동은 직접 보정하지 않고, 첫 관측 조건도 유지합니다. 메인 RTC처럼 미래 관측 채널에는 VJP가 전파될 수 있습니다. 실제 비동기 로봇 제어·지연 실험은 포함되지 않습니다.

전체 성공 판정은 세 블록이 목표 XY 반경6cm 안이고 모든 쌍의 Z 차이가3cm 초과인 순간입니다. 안정 유지 시간 기준은 없으며, `success_1/2`는 목표 XY 영역 진입 이력입니다. 단일 학습 seed의 ID 시뮬레이션입니다. 기존 공식 baseline 평가와 이 paired 비교는 난수 방식이 다르므로 각각의 baseline 수치를 섞지 않습니다. 데이터 출처와 제한은 [baseline 안내](../fm_baseline/results/stacking_baseline/README.md)를 참고합니다.

## 공개 전 확인

실제 공개 EMA·캐시와 고정 D3IL commit으로 GPU smoke 4회(2 contexts×2 arms×24 steps)를 실행했고 실패0회, ④ guidance40회, 관절 QP 허용오차와 LiPo 보정 한도 준수를 확인했습니다. 회귀 검사142개 통과·1개 건너뜀입니다. [검증 기록](stacking_combo31_smoke_20261116.json)을 보존했습니다. 이 짧은 검사는 실행 경로 검증이며 조합31의 전체 ID 성능 결과는 아직 없습니다.
