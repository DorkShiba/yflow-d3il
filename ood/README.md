# OOD 평가 조건 (Pushing·Inserting)

두 구현(`feature/ood-benchmark`의 `fm_pc/ood.py`, `ood_sas_v1`)을 하나로 합쳤다. **조건 강도는 구현 1, seed 규칙은 구현 2**를 따른다(`docs/PLAN.md` 3.1절).

| 요인 | 내용 | weak / mid / strong |
|---|---|---|
| A 블록 초기 위치 | 모든 블록을 시연 초기 범위 밖 d로, 가장 가까운 가능한 면부터 이동(결정적). strong은 팔 도달 고리(베이스에서 0.30–0.81 m) 안에만 두고, 다른 블록 시작 범위와는 5 cm만 띄운다(`generate.shift_blocks_far`) | 1 / 5 / 18 cm |
| B 로봇 시작 | 손을 시연 시작점에서 d만큼 seed 방향으로 이동. 시뮬레이터에서 블록 미이동(<1 mm)·도착 오차(<5 mm) 확인 | 2 / 5 / 10 cm |
| C 관측 노이즈 | 정책·실행 직전 필터가 보는 블록 x, y와 yaw에 함께 Gaussian. 실제 상태와 위반 측정에는 넣지 않음 | 2.5 mm·2.5° / 5 mm·5° / 10 mm·10° |
| D 실행 중 push | `push_step`(Pushing 80, Inserting 300) 이후 chunk 중간(남은 행동 4개)에 손에 가장 가까운 블록을 d만큼 순간이동 | 2 / 4 / 8 cm |
| E 추론 지연 | 남은 행동이 d개일 때 다음 계획을 세우고 새 chunk는 index d부터 실행(단순 비동기, RTC 논문의 baseline) | 1 / 2 / 4 step |

A 수치는 2026-11-12에 2.5/5/10 cm에서 1/5/18 cm로 바꿨다(데이터 v3). A_mid 배치는 v2와 같다. A_strong은 배치 후 시뮬레이터에서 2 s 동안 블록이 2 mm 안에 머무는지 확인하고, 안 되면 그 context만 1 cm씩 낮춘다(v3: Pushing test_028 17 cm·test_059 8 cm, Inserting env_011_04 17 cm). 실제 d는 `shift[].shift_m`에 있다. 블록별 이동 방향은 각 행의 `shift[].face`(x−/x+/y−/y+)에 있다. y+는 목표 쪽 이동이라 쉬워지므로 결과를 방향별로 나눠 보는 것이 좋다.

조건 20개: `id`, `{A,B,C}_{weak,mid,strong}`, `{AB,AC,BC,ABC}_mid`, `{D,E}_{weak,mid,strong}`. 손끝 yaw는 제외했다.

## seed 규칙

- `seed_for(*parts)` = SHA-256(`'|'.join(parts)`)의 앞 8바이트. 루트 seed는 `20261106`이다.
- 요인별 seed: `seed_for(root, task, context_id, 'B'|'C'|'D')`. 강도와 조합이 달라도 같은 context는 같은 방향·노이즈를 쓴다(예: `AB_mid`의 블록 배치 = `A_mid`, `BC_mid`의 시작 위치 = `B_mid`).
- 관측 노이즈 값: `gauss(seed_for(noise_seed, rollout, step, block, 'x'|'y'|'yaw'))`. 재계획 횟수·호출 순서와 무관하다.
- 정책 noise: 사례의 `policy_seeds`(4개, `seed_for(root, task, context_id, 'policy', j)`)를 rollout마다 `torch.Generator`에 넣는다.
- 단, 본실험(`fm_constraint.experiment run --ood`)은 seed를 직접 준다(0·1·2). 이때 FM noise는 ID 평가의 context ID(Pushing `test_007` → `7`)로 만들어 **ID 본실험과 같은 noise**가 되고, `policy_seeds`는 쓰지 않는다. 관측 노이즈(C)·push(D)는 사례 seed와 rollout 번호(0·1·2)로 정해진다.

## 사용

```bash
MUJOCO_GL=disable python -m ood.generate --d3il-root ../d3il      # data/ood/{pushing,inserting}.jsonl + manifest.json
python -m ood.verify                                              # 해시·개수·seed·조합 재사용 검사
MUJOCO_GL=disable python -m fm_constraint.eval_pushing --config configs/pushing.yaml --ood D_mid --projection
python -m ood.metrics --eval-dir fm_constraint/runs/<id> --label id --eval-dir fm_constraint/runs/<D_mid> --label D_mid
```

`--ood`를 주면 context·로봇 시작·노이즈·push·지연·정책 seed를 모두 파일에서 읽는다(본실험 runner는 정책 seed만 0·1·2로 바꾼다. 위 seed 규칙 참고). `eval.n_contexts`로 앞쪽 일부만, `eval.n_traj`로 `policy_seeds` 앞쪽 일부만 쓸 수 있다. 결과의 `results.json`에는 `ood`(파일 sha256, 조건, 요인)와 rollout별 `ood_events`(push 기록)가 남고, `rollouts.json`의 각 step에는 chunk 시작 여부 `seam`이 있다.

| 파일 | 역할 |
|---|---|
| `conditions.py` | 조건·강도·seed 규칙 |
| `apply.py` | A 배치·D push 기하, 평가 중 C·D·E 적용, JSONL 로드 |
| `generate.py` | D3IL에서 사례 생성 (B 시뮬레이터 검증 포함) |
| `verify.py` | 생성 데이터 검사 |
| `metrics.py` | seam·chunk 내부 jerk, push 직후 jerk, 성공률의 context bootstrap 비교 |

학습 관측과의 최근접 거리(구현 1의 `OOD dist`)는 계획 관측을 저장하지 않아 이번 통합에서 제외했다.
