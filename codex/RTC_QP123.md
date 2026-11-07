# Pushing·Inserting ④ RTC + ⑤ QP 고정 실험

각 태스크·조건에서 ① Projection, ② PF2MP, ③ terminal-plan VLSA를 비트 순서대로 000부터 111까지 비교한다. 기준군 000도 RTC와 QP를 사용한다. 기존 ⑤ 고정 실험과 동일하게 ③의 실행 직전 재검사는 끈다.

- 고정값은 horizon 16, 청크 실행 8, Euler 10, 동기 RTC 지연 0, beta 5, LiPo jerk QP 보정 한도 0.3 mm다. 첫 계획에는 RTC 참조가 없으며 이후에는 직전 QP 출력의 미실행 8개 행동을 참조한다. 매 스텝에서 RTC 가이던스와 PF2MP를 적용하고 Euler 갱신 뒤 투영한다. 계획 끝에 VLSA, QP 순서로 적용한다.
- 고정 조건 데이터는 master의 cba3f3f에서 가져온 Pushing 60개·Inserting 82개다. A는 블록 배치, B는 손 시작 위치, C는 정책 관측 노이즈다. C는 재계획당 한 번 샘플링하고 물리 상태로 실행 지표를 계산한다. 각 태스크의 manifest.json에 13개 OOD 조건 이름이 있다.
- 방법 간에는 같은 전역 context 인덱스·trajectory·seed로 정책 노이즈를 짝짓는다. 결과에는 조건 파일 SHA256, RTC 사용 여부, 계획별 rtc_guided, QP 전후 위반, 성공률과 jerk가 기록된다. full ID와 OOD 한 조건은 각각 1 seed에 4,544 rollout이다.

## GPU 워크스페이스에서 실행

프로젝트의 기존 D3IL GPU 환경과 HardFlow 체크아웃을 사용한다. 저장소 루트에서 다음을 실행한다. 두 경로는 실제 서버 위치에 맞춘다.

```bash
git switch codex/rtc-qp-method123
python -m pip install -r requirements-qp123.txt
export HARDFLOW_ROOT=/root/d3il-work/codex/HardFlow
export D3IL_ROOT=/root/d3il-work/d3il
```

ID smoke는 두 태스크 각각 3 contexts × 4회 × 8조합, 합계 192 rollout이다. 명령 끝에 `--dry-run`을 추가하면 GPU 없이 배정만 확인한다. 1 GPU인 경우 `--devices cuda:0`만 지정한다.

```bash
for task in pushing inserting; do
  python -m fm_constraint.run_qp123 --rtc --condition id --task "$task" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 --n-contexts 3 --n-traj 4 --seeds 0 \
    --hardflow-root "$HARDFLOW_ROOT" --d3il-root "$D3IL_ROOT" \
    --output-root "codex/qp123_runs/rtc45_smoke_${task}" || break
done
```

본실험은 한 번에 한 태스크·한 조건을 지정한다. 먼저 ID에서 아래 명령을 Pushing과 Inserting에 각각 실행한다. 이후 condition을 원하는 OOD 이름으로 바꿔 반복한다. 13개 전부를 실행할 경우 이름은 A_weak, A_mid, A_strong, B_weak, B_mid, B_strong, C_weak, C_mid, C_strong, AB_mid, AC_mid, BC_mid, ABC_mid다.

```bash
task=pushing
condition=id
python -m fm_constraint.run_qp123_budget --rtc --task "$task" --condition "$condition" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --max-hours 20 \
  --n-contexts 0 --n-traj 4 --seeds 0 \
  --hardflow-root "$HARDFLOW_ROOT" --d3il-root "$D3IL_ROOT" \
  --output-root "codex/qp123_budget_rtc45_${task}_${condition}"
```

예시 본실험은 1 seed다. 여러 seed를 사용할 경우 처음부터 동일하게 `--seeds 0 1 2`를 지정한다. budget 실행기는 완료한 context별 결과를 저장한다. `progress.json`이 stopped이면 같은 명령에 `--resume`을 추가한다. `--max-hours`는 이 호출의 시간 제한이며 실제 완료 시간은 smoke로 측정한다.
