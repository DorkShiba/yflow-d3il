# 팀원용 실행 안내

기존 실험 환경에서 평가한 `rollouts.npz`를 각자의 로컬 웹에 불러와 3D로 재생하고, Plain/CAPS를 비교하거나 MP4/GIF로 저장합니다. 재학습은 필요하지 않습니다.

## 기존과 달라지는 점

| 단계 | 기존 | 뷰어를 사용할 때 |
|---|---|---|
| 학습 | baseline/CAPS 학습 | 그대로 유지. 재학습 불필요 |
| 평가 코드 | 기존 평가 스크립트 | 관절 기록 기능이 추가된 평가 코드 사용 |
| 평가 명령 | 기존 명령 | 끝에 `--record-joints` 추가 |
| 평가 결과 | `results.json`, `rollouts.npz` | 같은 파일 생성. NPZ에 관절·블록 XYZ·목표 정보 추가 |
| 결과 확인 | 지표·궤적 확인 | 웹에서 3D 재생·메트릭 그래프·여러 셋업 비교 |
| 영상 제작 | 별도 작업 | 웹에서 MP4/GIF 저장 |

`train.py`만 실행하면 NPZ가 만들어지는 것은 아닙니다. 체크포인트를 사용하는 `eval_pushing.py` 또는 `eval_inserting.py` 평가가 끝나야 생성됩니다.

## 1. 수정된 코드 받기

작업 중인 실험 폴더와 섞이지 않도록 별도 디렉토리에 받으세요.

```bash
git clone --branch visualizer https://github.com/DorkShiba/yflow-d3il.git yflow-visualizer
cd yflow-visualizer
```

이 브랜치는 뷰어와 평가 기록 기능을 제공합니다. 별도 실험 브랜치의 CAPS 학습 코드를 대체하는 용도가 아닙니다. 기존 체크포인트를 평가하는 데 사용하세요. 평가 파일만 복사하면 `smoothness.py` 등 의존성이 빠질 수 있으므로 브랜치 전체를 받으세요.

## 2. 기존 환경에서 관절 기록을 포함해 평가하기

기존 D3IL·HardFlow·PyTorch 실험 환경을 사용합니다. 해당 환경이 있는 GPU 서버나 로컬 CPU에서 평가할 수 있습니다. **기존 평가 명령의 나머지 옵션을 유지하고 `--record-joints`를 추가하세요.** 실행 파일은 수정된 브랜치의 평가 코드여야 합니다.

아래는 Pushing 단일 rollout 확인 예시입니다. 저장소 최상위 디렉토리에서 실험용 Python 환경을 활성화하고, 경로를 자신의 환경에 맞게 바꾸세요.

```bash
python fm_baseline/eval_pushing.py \
  --run-dir /absolute/path/to/my_experiment \
  --ckpt last.pt \
  --d3il-root /absolute/path/to/d3il \
  --hardflow-root /absolute/path/to/HardFlow \
  --data-dir /absolute/path/to/pushing_sincos_v1 \
  --device cpu \
  --contexts test --n-contexts 1 --n-traj 1 \
  --execute-steps 8 --seed 0 \
  --record-joints
```

이 예시는 사용법 확인용입니다. 실제 비교에서는 기존 context 분할·개수·순서, seed, 실행 스텝, 최대 스텝, 노이즈 옵션과 device를 유지하세요. `--paired-noise`를 썼다면 양쪽 모두 유지합니다. Pushing CLI의 실행 스텝 기본값은 1이므로 기존 실험이 8이었다면 `--execute-steps 8`을 명시하세요.

Inserting도 기존 `eval_inserting.py` 명령과 `--split val` 등의 설정을 유지하며 `--record-joints`를 추가하면 됩니다.

체크포인트 평가에는 해당 모델의 다음 세 파일이 필요합니다.

```text
my_experiment/
├── last.pt          # 또는 baseline_ema.pt 등 실제 체크포인트 파일명
├── config.json
└── normalizer.npz
```

평가가 끝나면 터미널의 `saved:` 경로를 확인하세요. 기본 위치는 다음과 같고, `--output-root`를 지정하면 해당 경로 아래에 생성됩니다.

```text
my_experiment/
└── eval_pushing_…/ 또는 eval_inserting_…/
    ├── results.json
    └── rollouts.npz
```

관절 기록은 각 제어 스텝 실행 직후의 7개 관절각입니다. 과거 NPZ에 관절 기록이 없으면 손·블록 궤적만 재생할 수 있습니다. 팔까지 움직이려면 옵션을 붙여 체크포인트 평가를 다시 해야 합니다. 재학습은 필요하지 않습니다.

## 3. 결과 파일을 본인 PC로 가져오기

서버에서 평가했다면 `rollouts.npz`를 다운로드하세요. Plain/CAPS를 비교하려면 각각의 파일을 실험명으로 구분해 보관합니다.

**결과를 재생·비교하는 PC에는 `.pt`, D3IL, HardFlow, PyTorch 또는 GPU가 필요하지 않습니다.** 웹 실행 환경과 FFmpeg만 준비하면 됩니다. 브라우저 3D에는 WebGL 지원이 필요합니다.

## 4. 로컬 웹 실행하기

Python 3.10 이상과 PATH에서 실행 가능한 FFmpeg가 필요합니다. macOS에서 Homebrew를 사용한다면:

```bash
brew install ffmpeg
```

Ubuntu/Debian에서는 `sudo apt install ffmpeg`로 설치할 수 있습니다. 받은 저장소 최상위 디렉토리에서 실행합니다.

```bash
./visualizer/run.sh
```

첫 실행에는 `.venv-visualizer`와 웹용 패키지가 자동 설치됩니다. Node.js는 필요하지 않습니다. 터미널을 유지하고 브라우저에서 <http://127.0.0.1:8765>에 접속하세요. 각자 PC의 로컬 주소이며 공용 서버 주소가 아닙니다. 종료는 터미널에서 `Ctrl+C`입니다.

포트가 사용 중이면 `PORT=8766 ./visualizer/run.sh`로 실행하고 <http://127.0.0.1:8766>에 접속합니다. 코드 업데이트 후 패키지도 갱신하려면 다음을 실행합니다.

```bash
.venv-visualizer/bin/python -m pip install -r visualizer/requirements.txt
```

현재 실행 스크립트는 macOS/Linux용입니다. 더블클릭 실행기나 실험 환경 자동 설치 기능은 아직 없습니다.

## 5. NPZ 가져오기 → 비교 → 영상 저장

1. 기본 샘플로 재생을 확인합니다. 자기 결과만 보려면 **모두 비우기**를 누릅니다.
2. **데이터 · 실행 설정 → 궤적 가져오기**에서 `rollouts.npz`를 선택합니다.
3. 셋업 이름과 실제 제어 스텝 간격을 입력하고 **가져오기**를 누릅니다. 현재 기본 프로토콜은 0.035초이며, 실험이 다르면 실제 값을 입력하세요.
4. **작업과 결과**에서 context/trajectory를 고르고 **비교에 추가**합니다. 다른 셋업도 같은 방식으로 추가합니다.
5. 좌우·상하·2×2 배치를 선택합니다. 최대 4개를 공통 시간축으로 재생하고 그래프를 클릭해 원하는 시점으로 이동할 수 있습니다.
6. 3D 시점을 조정하고 저장할 시작·종료 시각과 재생 배속을 선택합니다.
7. **비교 영상 내보내기 → MP4/GIF → 현재 배치로 내보내기**를 누릅니다.
8. 완료되면 **영상 다운로드**로 저장합니다. 3D 프레임 생성 중에는 브라우저와 로컬 서버를 유지하세요.

MP4는 흰 배경의 H.264 형식으로 PowerPoint에 넣을 수 있습니다. 3D MP4는 패널당 640×480 / 30fps / 최대 60초, GIF는 패널당 480×360 / 15fps / 최대 15초입니다. 개별 영상은 패널 하나만 남겨 출력합니다. NPZ 업로드는 64 MiB, 압축 해제된 배열 합계는 256 MiB까지 지원합니다.

## 선택: 웹에서 체크포인트 평가하기

로컬 PC에도 실험 환경이 있다면 **데이터 · 실행 설정 → 체크포인트 실행**을 사용할 수 있습니다. `.pt` 하나만 업로드하는 방식은 아닙니다.

- `config.json`, `normalizer.npz`, `.pt`가 있는 실험 폴더와 파일명을 지정합니다.
- 실험용 Python 실행 파일, D3IL·HardFlow·전처리 data 경로를 지정합니다.
- task, context, seed 등의 조건을 지정하고 **CPU 작업 제출**을 누릅니다.

웹 평가는 `--record-joints`를 자동으로 추가합니다. 웹용 가상환경만 설치해서는 평가할 수 없습니다. 브라우저 3D 관절 재생에는 **로봇 영상 + 2D 궤적** 옵션이 필요하지 않습니다. 그 옵션은 MuJoCo 렌더러로 별도 영상을 녹화하는 기능입니다.

## 결과를 설명할 때

- Plain/CAPS는 동일 task·context·seed·평가 조건으로 비교하세요. 단일 rollout 성공은 전체 성공률이 아닙니다.
- 기본 Pushing OFF/CAPS 샘플은 원본 100k pilot 기록이며 관절이 없습니다. 기본 Inserting은 관절 기록이 있는 baseline 1M 결과이며 CAPS 결과가 아닙니다.
- 팔은 기록된 관절각을 재생하지만, 브라우저 장면은 원본 D3IL의 벽·도구 형상을 완전히 복제한 것은 아닙니다.
- 자세한 구현과 검증 범위는 [README.md](README.md)를 참고하세요.

## 팀원에게 전달할 짧은 안내

> 학습은 기존대로 하면 됩니다. visualizer 브랜치의 평가 코드로 기존 평가 명령에 `--record-joints`를 추가해 실행해주세요. 나온 `rollouts.npz`를 본인 PC로 가져와 `./visualizer/run.sh`로 실행한 웹에 넣으면 관절까지 움직이는 3D 재생과 비교 영상 저장이 가능합니다. 결과를 보기만 하는 PC에는 체크포인트나 D3IL 환경이 필요하지 않습니다.
