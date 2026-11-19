# Stacking 상대좌표 · 절대좌표 32조합

같은 train/validation split(1,001/93개), 학습 seed 42, 23차원 관측, 8차원 행동으로 FM 두 모델을 학습했다. 두 학습 모두 60만 step에서 조기 종료했다. 별도 ID dev 40 contexts × 2회에서 성공률이 가장 높고, 동률이면 EMA 검증 손실이 가장 낮은 45만 step을 두 모델 모두 선택했다.

- [상대좌표 모델](../../../fm_baseline/results/stacking_relative_20261118/)은 각 블록의 월드 XYZ에서 현재 입력의 **목표 관절각 7개**를 D3IL 순기구학에 넣어 얻은 TCP 월드 XYZ를 뺀다. 좌표축은 월드 축이며 블록 yaw의 sin/cos, 관절각과 그리퍼 폭도 입력에 들어간다.
- [절대좌표 모델](../../../fm_baseline/results/stacking_absolute_20261118/)은 같은 블록 XYZ를 월드 좌표로 입력한다. 두 모델 모두 쌓기 목표 위치를 별도 입력으로 받지 않으며, 출력은 관절 증분 7개와 다음 관측의 그리퍼 폭이다.
- 두 모델 모두 학습 때 저장한 자기 관측 프레임의 normalizer를 쓴다. EMA 가중치는 학습 체크포인트와 텐서 단위로 일치한다. 방법 ②의 7관절 PF 캐시와 두 normalizer의 관절 중심·스케일이 일치해 팀 실험의 기존 캐시를 공유한다. 방법 ①~⑤의 관절 제약과 설정도 동일하게 사용한다.

## 실행

[D3IL 및 Python 환경 설치](../../stacking_team_run.md)를 따른다. D3IL 커밋은 `1d9c71850d4fc1477cee17b557c8b97d70b13071`이다. 저장소에 추론용 EMA, 모델 설정, normalizer와 해시가 포함되어 있다. 학습 시연은 필요 없다.

```bash
# 두 모델에서 ID와 OOD의 baseline 및 전체 방법 조합을 짧게 확인
for model in absolute relative; do
  bash scripts/run_stacking_experiment.sh --model "$model" --name smoke \
    --cond id,A_3 --combos 0,31 --contexts 0 --smoke 16
done

# GPU 한 대에서 모델별 32조합 전체를 순차 평가
for model in absolute relative; do
  bash scripts/run_stacking_experiment.sh --model "$model" --name team \
    --cond id,A_weak,A_3,A_mid,A_strong --combos all --contexts 0-29
done

# 각 모델과 조건의 모든 팀원 보고서가 모이면 통합
for model in absolute relative; do
  for cond in id A_weak A_3 A_mid A_strong; do
    python -m fm_constraint.stacking_experiment merge --model "$model" --cond "$cond"
  done
done
```

GPU에 작업을 나눌 때는 `--contexts 0-7`처럼 범위를 나누고 같은 context의 32조합은 같은 장치에서 실행한다. 필요한 경우 `--d3il-root`에 D3IL 체크아웃 경로를 지정한다. 모델별 원본 결과는 `experiments/stacking_v1_<model>_20261118/`, 보고서는 `docs/stacking/<model>_20261118/`에 기록된다. 두 모델의 비교는 같은 조건·조합·context·noise seed로 짝을 맞춰 수행한다.

- 조합 번호는 ① projection=16, ② PF2MP=8, ③ 실행 필터=4, ④ RTC=2, ⑤ LiPo=1의 합이다. S00은 FM baseline, S31은 다섯 방법 모두 활성화한다.
- [고정 ID 30개](../../../configs/stacking_experiment_contexts.json)와 [같은 index의 OOD A 4조건](../../../data/ood_stacking/manifest.json)을 사용한다. 각 context는 seed 0·1·2로 반복하며 조합마다 같은 hash 기반 CPU noise 초기값을 사용한다. 최대 1,000 step, plan 16, execute 8, Euler 10, RTC 지연 0이다. 모델당 14,400회, 두 모델 합계 28,800회다.
- 성공 판정은 세 블록의 목표 XY 반경 6 cm 이내와 모든 블록 쌍의 Z 높이 차이 3 cm 초과다. 블록 사이 접촉이나 안정 유지 시간은 검사하지 않는다. 손끝 jerk는 시뮬레이터 TCP의 RMS jerk이며, 청크 경계 값은 실행 명령의 관절 속도 변화량이다.
- [앞서 실행한 S00 비교 900회](s00_summary.json)의 `baseline`은 이번 45만 step 절대좌표 모델이다. 이전 S00 평가는 샘플 순번 기반의 장치 RNG를 사용했다. 32조합에서는 팀 러너의 context ID 기반 CPU RNG를 써 두 모델에서 S00도 새로 평가한다.
- 기존 35만 step 팀 실험은 `--model legacy`(기본값)로 읽고 집계할 수 있다. 기존에 시작한 legacy 원본 단위를 이어서 실행할 때는 해당 결과의 `git_commit`으로 실행한다. 실행기는 코드 fingerprint가 다른 원본 단위를 혼합하지 않는다.
