# D3IL 데이터 및 실험 환경 인수인계

작성일: 2026-10-11

이 문서는 사용자가 VESSL 터미널에서 실행하고 공유한 결과를 바탕으로 작성했다. 아래 `/root/...` 경로는 **VESSL 워크스페이스 경로**다. 이 README가 있는 로컬 Windows 폴더에 데이터가 저장된 것은 아니다.

## 1. 현재 상태

실험 대상은 **Pushing과 Inserting**이다. 입력은 위치·방향 등 상태 정보이며 이미지를 사용하지 않는다. 초기에 준비한 Stacking 데이터는 보관 중이며 현재 실험 대상에서 제외한다.

- 완료: 원본 다운로드, 필요한 상태 파일 추출, 원본 검사, 전처리, 학습·검증 분할, 저장 파일 검증.
- 완료: 두 태스크의 시뮬레이터 생성·시작·초기화 및 자세 유지 명령으로 기본 step 실행.
- 미완료: 모델 로더 연결, 정규화, 관측/행동 시퀀스 구성, 평가용 sin/cos 관측 구성, 행동 변환 연결, 실제 학습·작업 성공률·제약 위반율 평가.

**데이터가 준비됐다는 것은 모델 학습과 평가 파이프라인까지 완성됐다는 뜻이 아니다.** 현재 시뮬레이터 검사는 자세 유지 명령에 대한 기본 실행 확인이며, 시연 재현이나 학습 모델의 작업 성공은 검증하지 않았다.

## 2. 출처와 저장 위치

- 공식 코드: <https://github.com/ALRhub/d3il>
- 공식 프로젝트: <https://alrhub.github.io/d3il-website/>
- 논문: <https://arxiv.org/abs/2402.14606>
- 공식 README에서 연결한 데이터 ZIP: <https://drive.google.com/file/d/1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8/view>

실제 확인한 ZIP 크기는 약 16.71 GB, 전체 압축 해제 크기는 약 17.69 GB이며 내부 항목은 4,411,562개다. 전체 이미지를 풀지 않고 필요한 상태 파일만 추출했다. ZIP 최상위에는 aligning, avoiding, inserting, pushing, sorting, stacking이 있었고 Arranging은 없었다.

```text
/root/d3il-work/
├── downloads/data.zip
├── d3il/                         # 공식 저장소 + Inserting import 수정
├── raw/
│   ├── pushing/all_data/         # 2,000개 pkl, 약 327 MB
│   ├── inserting/data_3/         # 800개 pkl, 약 655 MB
│   └── stacking/all_data/        # 보관용
└── codex/
    ├── venv/                    # 작업용 Python 가상환경
    ├── constraints.txt
    ├── processed/
    │   ├── pushing_sincos_v1/
    │   ├── inserting_sincos_v1/
    │   └── stacking_sincos_v1/   # 보관용
    ├── backups/                 # 수정 전 원본 코드 백업
    └── ...                      # 검사·전처리 스크립트 및 보고서
```

공식 로더 경로에는 다음 심볼릭 링크를 연결했다.

```text
d3il/environments/dataset/data/pushing/all_data
  -> /root/d3il-work/raw/pushing/all_data
d3il/environments/dataset/data/stacking/all_data
  -> /root/d3il-work/raw/stacking/all_data
```

Inserting은 위와 같은 공식 로더 연결을 만들었다고 확인된 상태가 아니다. 전처리본은 raw 경로에서 직접 생성했다.

`/root`는 약 100 GB의 별도 마운트로 확인했다. **워크스페이스 삭제 후 영구 보존되는지는 확인하지 않았으므로 VESSL 스토리지 설정에서 확인해야 한다.** tmux는 터미널 연결 종료에 대응하는 도구이며 워크스페이스 중지·재시작으로부터 프로세스를 보존하지 않는다.

## 3. 데이터 규모와 분할

| 항목 | Pushing | Inserting |
|---|---:|---:|
| 태스크 | 두 블록을 목표 구역으로 밀기 | 장벽이 있는 공간에서 세 블록을 목표 구역으로 밀어 배치하기 |
| 전체 시연 | 2,000 | 800 |
| 학습 시연 | 1,800 | 718 |
| 검증 시연 | 200 | 82 |
| 학습 상태·행동 쌍 | 417,518 | 711,908 |
| 검증 상태·행동 쌍 | 45,868 | 80,479 |
| 전체 상태·행동 쌍 | 463,386 | 792,387 |
| 원본 시점 수 합계 | 465,386 | 793,187 |
| 시연 길이 최소/중앙값/최대 | 152 / 230 / 356 | 533 / 952 / 2,355 |
| 기록 간격 | 0.035초 | 0.035초 |
| 입력/행동 차원 | 12 / 2 | 16 / 2 |

시연이 T시점이면 상태·행동 쌍은 T−1개다. 마지막 상태에는 다음 시점의 행동 표적을 만들 수 없기 때문이다.

### Pushing: 공식 분할

`d3il/environments/dataset/data/pushing/train_files.pkl`과 `eval_files.pkl`을 사용했다. 누락·분할 내 중복·분할 간 동일 파일·분할되지 않은 원본은 모두 0개였다.

`train_contexts.pkl`과 `test_contexts.pkl`은 각각 60개다. 이는 시뮬레이터 초기 조건이며, 검증 시연 200개와 구분한다. 이번 검사는 **동일 파일의 중복을 확인한 것이며, 공식 분할의 모든 유사 초기 조건이 분리됐다고 보장하지 않는다.**

### Inserting: 자체 분할

전용 공식 학습 로더·분할·평가 연결 코드는 확인한 폴더에서 찾지 못했다. 따라서 아래 방식으로 자체 분할했다.

1. 각 시연의 context를 숫자 배열로 펼친다.
2. 소수 6자리 반올림 후 바이트의 SHA-256으로 그룹을 만든다.
3. 그룹 키를 정렬하고 `np.random.default_rng(42)`로 섞는다.
4. 793개 그룹 중 `ceil(793 × 0.1) = 80`개를 검증에 배정한다.
5. 동일 그룹의 시연은 같은 분할에 배치한다.

그룹 786개는 시연 1개, 7개는 시연 2개를 포함한다. 결과는 학습 718개·검증 82개이며 그룹 중복은 없다. **공식 분할이 아니고, 물리적으로 동등한 초기 조건이나 모든 근접 초기 조건의 분리까지 보장하는 방식도 아니다.** 재분할 없이 manifest를 공유해서 실험 간 분할을 고정한다.

## 4. 입력과 행동의 정확한 의미

### 공통 개념

- `des_c_pos`: 제어기가 추종하도록 지정한 목표 손 위치.
- `c_pos`: 실제 로봇 손 위치. 목표 위치와 다를 수 있다.
- `yaw`: 블록의 회전 방향 중 z축 회전각. 원본 쿼터니언에서 공식 `geo_transform.py`의 `quat2euler`로 구한다.
- 행동은 **목표 손 XY의 시점 간 변화량**이다. 절대 좌표, 실제 손 변위, 시간으로 나눈 속도와 구분한다.

```python
action[t] = des_c_pos[t + 1, :2] - des_c_pos[t, :2]
```

### Pushing 입력 순서: 12차원

| 인덱스 | 값 |
|---|---|
| 0:2 | 목표 손 x, y |
| 2:4 | 실제 손 x, y |
| 4:8 | 빨간 블록 x, y, sin(yaw), cos(yaw) |
| 8:12 | 초록 블록 x, y, sin(yaw), cos(yaw) |

공식 로더의 10차원 구성에서 두 블록의 방향 표현을 각각 1→2차원으로 변경했다. 공식 로더처럼 목표 구역 위치는 모델 입력에 넣지 않았다. 목표 위치를 가변 조건으로 바꾸는 실험에서는 입력 구성을 재검토해야 한다.

### Inserting 입력 순서: 16차원

| 인덱스 | 값 |
|---|---|
| 0:2 | 목표 손 x, y |
| 2:4 | 실제 손 x, y |
| 4:8 | box-1의 x, y, sin(yaw), cos(yaw) |
| 8:12 | box-2의 x, y, sin(yaw), cos(yaw) |
| 12:16 | box-3의 x, y, sin(yaw), cos(yaw) |

이 16차원은 **팀에서 구성한 입력 규격**이다. 원본 전체 검사에서 다음 값은 고정이었다.

- 목표 손 높이: `0.12235930599088184`
- 목표 손 방향 쿼터니언: `[0, 1, 0, 0]` (w, x, y, z 순서)
- 목표 구역 위치: `[0.3575, 0.276, 0]`, `[0.525, 0.4535, 0]`, `[0.6925, 0.276, 0]`
- 시연 XY 행동 범위: 각 축 대략 −0.00833359 ~ 0.00833333

마지막 범위는 관측된 데이터 범위이지, 물리적 안전 한계나 확정된 클리핑 기준이 아니다. 목표 위치와 장벽 배치를 변경한다면 현재 입력만으로 변경된 환경을 충분히 설명할 수 있는지 검토해야 한다.

## 5. tan 대신 sin/cos를 사용한 이유

Stacking 준비 과정에서 공식 `tan(yaw)`가 약 `1.63e16`까지 커지는 것을 확인했다. 유한한 값이라 NaN/Inf 검사에는 통과하지만, 정규화와 학습에 극단값으로 작용할 수 있다. 이에 두 실험 태스크도 다음처럼 변경했다.

```text
기존: [tan(yaw)]
변경: [sin(yaw), cos(yaw)]
```

sin을 cos로 나누는 것이 아니다. 두 값을 각각 입력하며 범위는 −1~1이다. sin/cos는 tan과 달리 180도 차이를 구별하므로 표현의 정보와 대칭성도 달라진다. 따라서 공식 표현과 완전히 동일한 설정이라고 보고하지 않는다.

베이스라인과 제약 적용 모델 모두 같은 표현을 사용해야 한다. 학습·검증·실행 평가에서 각도 계산 방식과 항목 순서도 동일해야 한다.

## 6. 전처리 파일 형식과 읽기

각 태스크 폴더에는 `train/`, `val/`, `manifest.json`이 있다. 시연 하나를 NPZ 파일 하나로 저장한다.

| 키 | shape | dtype / 의미 |
|---|---|---|
| `obs` | `(T-1, D)` | float32, 현재 상태 |
| `actions` | `(T-1, 2)` | float32, 목표 XY 변화량 |
| `next_obs` | `(T-1, D)` | float32, 다음 상태 |
| `timestamps` | `(T-1,)` | 원본 시각 유지 |
| `next_timestamps` | `(T-1,)` | 다음 시각 유지 |

```python
from pathlib import Path
import json
import numpy as np

root = Path('/root/d3il-work/codex/processed/pushing_sincos_v1')
manifest = json.loads((root / 'manifest.json').read_text())
record = manifest['splits']['train'][0]

with np.load(root / record['file'], allow_pickle=False) as data:
    obs = data['obs'].copy()
    actions = data['actions'].copy()
print(obs.shape, actions.shape)
```

`manifest.json`이 입력 항목 순서, 행동 정의, 분할, 원본 파일 매핑의 기준이다. NPZ에 context 전체를 복사하지 않았으므로 시연 초기 조건 재현에는 원본 pkl 또는 별도 평가 context가 필요하다. 원본을 보관한다.

**아직 적용하지 않은 처리:** 정규화, 패딩, 시퀀스 길이 고정, 행동 클리핑, 제약 라벨 생성. 한 시연의 끝과 다른 시연의 시작을 이어서 학습 구간을 만들면 안 된다.

## 7. 수행한 검증과 한계

- 전체 원본의 항목 구성, 시간축 길이, 숫자 배열의 NaN/Inf, 기록 간격 확인.
- Pushing·Inserting 원본 검사에서 보고된 문제 0개.
- 전처리본 차원과 숫자 유효성 확인.
- sin²(yaw) + cos²(yaw) ≈ 1 확인.
- 현재 목표 XY + 행동 ≈ 다음 목표 XY 확인.
- 저장 파일 재열람 검사 및 시연·상태/행동 쌍 개수 확인.
- Inserting 자체 분할의 context 그룹 중복 없음 확인.

이는 배열 구성과 저장의 검증이다. 학습 적합성, 일반화 성능, 시연을 명령으로 재생했을 때의 재현성, 제약 적용 효과를 보장하지 않는다.

Stacking 원본의 `des_c_pos`, `des_c_quat`에는 각 1,095건의 NaN/Inf 보고가 있었으나, 해당 태스크 전처리는 목표 관절 위치를 사용하므로 최종 배열에서는 문제가 없었다. 이 사실을 Pushing·Inserting의 결측 문제로 혼동하지 않는다.

## 8. VESSL 실행 환경

공유된 실행 결과 기준:

| 항목 | 값 |
|---|---|
| Python | 3.10.12 |
| 워크스페이스 GPU | RTX 3090 24 GB × 2 |
| PyTorch | 2.3.1+cu121 |
| torchvision | 0.18.1+cu121 |
| NumPy | 1.26.4 (venv 내부) |
| MuJoCo | 2.3.2 |
| Gym | 0.23.1 |
| Pinocchio | 2.6.21 (`pip` 패키지명 `pin`) |
| OpenCV | import 시 4.10.0 표시 |

가상환경은 `--system-site-packages`로 생성하여 기존 PyTorch 등을 공유한다. **완전 독립 환경은 아니므로 기본 컨테이너의 변경이 영향을 줄 수 있다.** 공식 install.sh의 PyTorch 1.13 / Gym 0.21 구성과도 다르다.

SciPy 1.10.1, Matplotlib 3.7.5, gin-config, pyyaml, tqdm, imageio 등을 설치하도록 안내했으며 PyBullet은 import 성공을 확인했다. 정확한 전체 버전은 실제 환경에서 기록한다. Pinocchio 의존성이 NumPy 1.26을 요구해 초기 1.24.4 고정을 변경했다. OpenCV import에 필요한 시스템 `libgl1`도 설치했다.

### 새 터미널에서 활성화

```bash
source /root/d3il-work/codex/venv/bin/activate
cd /root/d3il-work/d3il
export PYTHONPATH="/root/d3il-work/d3il:/root/d3il-work/d3il/environments/d3il${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=disable
```

`MUJOCO_GL=disable`은 렌더링을 끄는 설정이며 PyTorch GPU 학습을 끄지 않는다. 환경 생성 시 `render=False`를 사용했다. 이미지 렌더링·실행 영상 저장은 별도 설정과 검사가 필요하다.

### 환경 및 코드 수정 기록

아래 명령은 이전에 안내했지만 실행 결과는 공유되지 않았다. 파일 존재 여부를 확인하고 필요 시 실행한다.

```bash
mkdir -p /root/d3il-work/codex/environment
python -m pip freeze > /root/d3il-work/codex/environment/packages.txt
git -C /root/d3il-work/d3il rev-parse HEAD > /root/d3il-work/codex/environment/d3il_commit.txt
git -C /root/d3il-work/d3il diff > /root/d3il-work/codex/environment/d3il_changes.patch
```

pip 목록만으로 GPU 드라이버·시스템 라이브러리·기본 컨테이너까지 복원되지는 않는다. VESSL 이미지 식별자와 스토리지 설정도 함께 기록하는 것이 좋다.

## 9. 수정한 코드와 환경의 알려진 문제

### Inserting import 경로 수정

다음 두 파일에서 `from d3il_sim.`을 `from environments.d3il.d3il_sim.`으로 통일했다.

```text
environments/d3il/envs/gym_inserting_env/gym_inserting/envs/gate_insertion.py
environments/d3il/envs/gym_inserting_env/gym_inserting/envs/objects/gate_insertion_objects.py
```

동일한 Box 클래스가 서로 다른 모듈 경로로 로드되어 `AttributeError: 'Box' object has no attribute 'mj_load'`가 발생했었다. 수정 후 초기화와 step 검사를 통과했다.

확인된 백업 경로:

```text
/root/d3il-work/codex/backups/20261010_050545_241483
```

### 환경 관측과 모델 입력의 차이

| 태스크 | 기존 환경 실제 관측 | 선언된 observation_space | 전처리 모델 입력 |
|---|---:|---:|---:|
| Pushing | 8 | 14 | 12 |
| Inserting | 11 | 14 | 16 |

기존 환경 관측은 실제 손 XY와 블록별 XY·tan(yaw)를 포함한다. 모델 입력에는 목표 손 XY가 추가되고, 블록 방향도 sin/cos로 바뀐다. **env.reset()/step()의 관측을 그대로 모델에 넣거나 선언된 observation_space로 모델 차원을 정하면 안 된다.**

평가용 입력은 실제 로봇 상태와 물체 쿼터니언에서 다시 구성해야 한다. 기존 tan 값에 arctan을 적용해 sin/cos를 복구하면 180도 차이를 잃으므로, 원본 쿼터니언에서 동일한 함수로 계산한다.

### 반환 시점 점검 필요

확인한 공통 GymEnvWrapper.step 코드는 관측·보상·종료 여부를 물리 시뮬레이션의 next_step 루프 전에 계산한다. 이후 계산되는 일부 info와 시점이 다를 수 있다. 평가 연결 시 실제 흐름을 확인하여 최신 관측, reward, done, info의 시점을 일치시켜야 한다. 기본 step 검사만으로 이 문제가 해결된 것은 아니다.

## 10. 모델 및 평가 코드와 합의할 사항

1. 과거 관측 길이와 미래 행동 예측 길이.
2. 배치 형태와 시연 끝의 패딩·마스크 처리.
3. 상태·행동 정규화 방식과 적용 위치.
4. 학습 데이터만으로 통계를 계산하고 검증·평가에 동일 통계를 적용하는 방법.
5. 출력 행동의 정규화를 되돌린 뒤 물리 단위로 제약을 적용하는 순서.
6. 여러 미래 행동 중 몇 개를 실행한 뒤 다시 관측할지.
7. 환경 초기화 직후 이전 목표 XY를 어떤 값으로 초기화할지.
8. 초기 조건과 평가 seed, 에피소드 길이 제한, 성공·제약 위반의 정의.

개념적인 제어 연결은 다음과 같다.

```text
환경의 최신 상태 + 이전 목표 XY
  -> 학습과 동일한 sin/cos 관측 구성
  -> 학습 시 정한 정규화
  -> 모델 예측
  -> 행동 정규화 역변환
  -> 다음 목표 XY = 이전 목표 XY + 예측 변화량
  -> 목표 높이·쿼터니언 추가
  -> 환경에 7차원 위치·자세 명령 전달
```

두 모델의 행동 출력은 2차원이어도 시뮬레이터에 전달하는 명령은 7차원이다. 높이·방향 선택과 제어 간격은 데이터 수집 설정에 맞춰야 한다. Inserting은 고정 목표 높이를 확인했지만, Pushing은 전체 시연의 목표 높이·방향 범위까지 별도로 집계하지 않았다. 공식 Pushing 평가 코드는 초기 실제 높이를 유지하고 방향 `[0,1,0,0]`을 사용한다. 데이터와의 정합성은 추가 확인한다.

Pushing의 n_substeps는 실행에서 35로 확인됐고, Inserting 생성자 기본값도 35다. 물리 dt와 함께 실제 제어 간격을 확인하여 데이터 기록 간격 0.035초와 맞춘다.

Inserting 환경 기본 에피소드 제한은 코드상 2,000인데 시연은 최대 2,355시점이다. 이를 이유로 데이터를 임의로 잘라내지 않는다. 재생·평가 시 적절한 제한을 팀에서 정하고 비교 실험 모두에 동일하게 적용한다.

## 11. 작업 스크립트

모두 VESSL의 `/root/d3il-work/codex/` 아래에 있다.

| 파일 | 역할 |
|---|---|
| `audit_data.py` | 초기 Stacking·Inserting 원본 검사 |
| `audit_data_with_pushing.py` | Pushing 포함 원본 검사 |
| `data_audit_with_pushing.json` | 원본 검사 보고서 |
| `prepare_stacking.py` | 보관 중인 Stacking 전처리 |
| `inspect_inserting.py` | 높이·방향·목표 위치·context 검사 |
| `prepare_inserting.py` | 자체 그룹 분할과 Inserting 전처리 |
| `prepare_pushing.py` | 공식 분할과 Pushing 전처리, 저장 후 검사 |
| `verify_processed.py` | Stacking·Inserting 전처리본 검사 |

`verify_processed.py`에는 현재 Pushing이 포함되지 않는다. Pushing은 prepare_pushing.py 내부에서 저장 후 검증했다. 전처리 스크립트는 출력 폴더가 존재하면 덮어쓰기를 막는다. 재실행 시 FileExistsError가 나면 삭제부터 하지 말고 manifest와 파일을 확인한다.

## 12. 팀원에게 전달할 요약

> D3IL Pushing·Inserting의 상태 기반 데이터 준비와 저장 검증을 완료했다. 입력은 각각 12·16차원, 행동은 모두 목표 손 XY 변화량 2차원이며 방향은 sin/cos다. 시연별 NPZ와 manifest로 저장했고 정규화·패딩·시퀀스 구성은 하지 않았다. Pushing은 공식 분할, Inserting은 자체 context 그룹 분할을 사용한다. 두 시뮬레이터는 초기화와 자세 유지 step 검사까지 통과했다. 모델 로더와 평가용 관측·행동 변환, 반환 시점 정합성 및 실제 작업 성능 검증은 남아 있다.

