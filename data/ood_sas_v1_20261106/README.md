# D3IL OOD 평가 초기 조건 v1

Sorting-2/4/6, Stacking, Avoiding의 **초기 조건·관측 노이즈 평가 명세**입니다. 전문가 시연, 학습용 행동 라벨, 실행 영상은 포함하지 않습니다. 공식 D3IL OOD 릴리스가 아니라 이 프로젝트에서 생성한 후보입니다.

## 파일과 규모

| 파일 | 사례 수 | 조건 수 |
|---|---:|---:|
| sorting2.jsonl | 1,560 | 13 |
| sorting4.jsonl | 1,560 | 13 |
| sorting6.jsonl | 1,560 | 13 |
| stacking.jsonl | 1,920 | 16 |
| avoiding.jsonl | 840 | 7 |

총 7,440개 명세이며 조건마다 dev 20개 / test 100개입니다. 사례별 정책 난수 4개는 예정 반복 평가용이지 이미 실행한 rollout을 뜻하지 않습니다. 생성 seed는 20261106입니다.

각 JSONL 행의 주요 필드:

- `id`, `base_id`, `split`, `task`: 사례/공통 기본 배치/분할/태스크.
- `blocks`: 이름, XY(m), yaw(rad), 보수적인 충돌 검사 반경.
- `robot_start`: canonical reset 대비 손끝 XY(m), yaw(rad) 오프셋. 관절 상태는 환경에서 IK로 계산해야 합니다.
- `noise`: 정규화 전 정책 관측에 더할 Gaussian 잡음의 종류·축별 표준편차·seed.
- `policy_seeds`, `repeats`: 방법 간 맞춰 사용할 정책 난수와 반복 횟수.
- `validation`: 생성 당시 검사 상태. 원본 보존을 위해 `rollout_ready=false`를 유지합니다.

## 조건

| 변화 | Sorting | Stacking | Avoiding |
|---|---|---|---|
| A 물체 위치 | 초기 영역 합집합 밖 3/6/10cm | 색상별 영역 밖 1.5/3/6cm | 없음 |
| B 손끝 XY | 2/5/10cm, 8방향 | 2/5/10cm, 8방향 | X ±2/5/10cm |
| B 손끝 yaw | 없음 | ±5/10/20° | 없음 |
| C 관측 XY σ | 블록 3/6/12mm | 블록 3/6/12mm | 실제 손 2.5/5/10mm |
| C 관측 yaw σ | 블록 2.5/5/10° | 블록 2.5/5/10° | 없음 |

A/B/C는 각각 독립 조건입니다. A는 한 블록만 옮깁니다. yaw 분포는 물체의 원래 ±90°를 유지합니다. ID도 보수적인 블록 간격 검사를 거쳐 생성하므로 공식 ID 테스트셋과 동일하지 않습니다.

## 검증 상태와 사용 시 주의

`manifest.json`의 생성 당시 상태와 `verification.json`은 구조·해시·기하 조건 검사만 나타냅니다. 이후 실제 MuJoCo 초기화 스크리닝에서 **7,313개 통과 / 127개 탈락**했습니다. 별도 `validation/`에 통과 ID, 탈락 이유, 원본 검사 로그의 SHA256을 보존했습니다. 원본 후보를 삭제하거나 경계로 clipping하지 않았습니다.

**초기화 통과는 정책 실행 가능성, 경로 존재, 태스크 성공을 보장하지 않습니다.** 스크리닝은 로봇 관절을 고정한 물체 안정화 검사이며, 컨트롤러·관측 이력 초기화와 실제 학습 분포 대조가 추가로 필요합니다. 사용 환경에서 다시 확인하고, 방법 간 동일한 유효 사례 목록을 고정하세요. 이 패키지에는 이후 Sorting-2 전용 controller 검증이나 50개 예비실험의 결과/선정 목록을 포함하지 않습니다.

관측 잡음은 정책 입력 사본에만 적용하고 실제 물체·로봇 상태, 명령 누적 기준점, 성공 판정, jerk 측정에는 넣지 마세요. Gaussian σ는 최대 오차가 아닙니다. `ood.generate.noise_sample`로 사례·반복·재계획·엔티티·축별 잡음을 재현할 수 있습니다. Stacking의 관절/그리퍼 행동은 XY 행동과 다르므로 별도 adapter가 필요합니다.

## 재현

저장소 루트에서 Python 표준 라이브러리만으로 실행할 수 있습니다.

```bash
python3 ood/verify.py data/ood_sas_v1_20261106
python3 -m unittest discover -s tests -p test_ood_generation.py
python3 ood/generate.py --output /tmp/ood_sas_reproduced
```

생성기는 기존 출력 폴더를 덮어쓰지 않습니다. `manifest.json`의 JSONL SHA256으로 원본 일치를 확인하세요. 시뮬레이터 검사는 `ood/validate_sim.py`를 사용하며 D3IL, NumPy, SciPy, MuJoCo 및 환경 설정이 별도로 필요합니다.

## 출처

환경 초기 분포와 구조: [공식 ALRhub/d3il](https://github.com/ALRhub/d3il/tree/1d9c71850d4fc1477cee17b557c8b97d70b13071), commit `1d9c71850d4fc1477cee17b557c8b97d70b13071`. 공식 시연 파일 자체는 이 OOD 패키지에 재배포하지 않습니다.
