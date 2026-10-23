# D3IL Rollout Viewer — local CPU visualizer

각 팀원이 자신의 컴퓨터에서 웹 서버를 실행합니다. 클라우드 계정·공용 서버·GPU는 필요하지 않습니다. 서버는 기본 `127.0.0.1`에만 바인딩하며, 체크포인트와 결과는 로컬에 남습니다.

팀원용 실행 순서와 기존 방식의 차이는 [TEAM_GUIDE.md](TEAM_GUIDE.md)를 참고하세요.

## 시작

```bash
./visualizer/run.sh
# 브라우저: http://127.0.0.1:8765
```

Python 3.10 이상이 필요합니다. 첫 실행은 `.venv-visualizer`를 만들고 웹/2D 의존성을 설치합니다. 영상 출력에는 PATH에 FFmpeg가 필요합니다(macOS `brew install ffmpeg`, Debian/Ubuntu `sudo apt install ffmpeg`). 포트 변경은 `PORT=8766 ./visualizer/run.sh`입니다. 기존 환경을 업데이트하려면 `.venv-visualizer/bin/python -m pip install -r visualizer/requirements.txt`를 실행합니다.

웹 UI는 빌드 없는 HTML/CSS/JavaScript, API는 FastAPI입니다. Node 설치 없이 사용할 수 있도록 프런트엔드 빌드 의존성을 제외했습니다. 작업은 한 번에 하나씩 처리하며 JSON 작업 기록과 결과는 `.visualizer/`에 저장합니다. `VISUALIZER_STORE`로 저장 위치를 바꿀 수 있습니다. 서버 재시작 시 진행 중이던 작업은 `interrupted`로 표시하며 자동 재실행하지 않습니다.

## 기존 결과 비교

1. 기존 평가의 `rollouts.npz`를 가져오고 이름과 실제 제어 스텝 간격을 입력합니다. 현재 프로토콜의 기본값은 0.035초입니다.
2. 작업 결과에서 context/trajectory key를 고르고 **비교에 추가**합니다.
3. 최대 4개를 좌우·상하·2×2로 배치합니다. `←`로 순서를 바꿀 수 있습니다.
4. 공통 시간축으로 재생·탐색·배속·스텝 이동합니다. 더 짧은 rollout은 마지막 상태를 유지합니다.
5. 구간과 형식을 선택해 비교 영상을 출력합니다. 하나의 패널만 남기면 개별 영상입니다.

MP4는 패널당 640×480 / 30fps, GIF는 패널당 480×360 / 15fps입니다. GIF 출력 길이는 15초, 선택 구간 끝은 120초 이내입니다. 출력 속도는 재생 배속을 따릅니다. 영상과 설정 JSON을 각각 다운로드할 수 있습니다. 2D 패널은 공통 XY 범위와 동일 축척을 사용합니다. 목표 손과 실제 손을 구분하며, 목표 위치가 있는 데이터에는 목표 허용 반경을 표시합니다. 메타데이터가 없는 임의 NPZ에는 목표를 추측해서 넣지 않습니다. 장애물 지도는 포함하지 않습니다.

저장된 배열은 **각 제어 스텝 실행 직후** 상태입니다. 첫 프레임은 첫 post-step 상태로 유지하며, 리셋 순간의 전체 상태를 복원했다고 해석하면 안 됩니다. 과거 NPZ에는 관절 상태가 없으므로 과거 로봇 전체 자세를 재현하지 않습니다. NPZ 업로드는 64 MiB, 압축 해제 합계는 256 MiB로 제한합니다.

## 체크포인트 실행

웹용 가상환경과 별도로 **기존 D3IL 실험이 실행되는 Python 환경**을 사용합니다. UI에서 해당 Python 실행 파일, D3IL / HardFlow / 전처리 data 경로를 지정합니다. 이 저장소의 루트 README와 `fm_baseline/README.md`의 실험 환경 구성을 따르세요. 해당 환경에 Pillow도 설치해야 로봇 프레임을 저장할 수 있습니다. 웹용 환경만 설치해서는 시뮬레이터가 실행되지 않습니다.

`config.json`, `normalizer.npz`, 가중치가 있는 run 폴더를 지정합니다. `fm_baseline/results/*`, `fm_baseline/runs/*`의 가중치는 자동 발견하며 외부 폴더는 직접 입력합니다. **본인이 신뢰하는 체크포인트만 사용하세요**. 기존 평가기가 PyTorch 체크포인트와 D3IL context pickle을 로드합니다.

- 모든 평가 명령은 `--device cpu`, `--paired-noise`, 명시적 실행 스텝을 사용합니다.
- Pushing: train/test 앞 N개 context, fresh/fixed 노이즈 지원.
- Inserting: 저장된 val context 앞 N개, fresh 노이즈만 지원. fixed는 지원되지 않는 옵션으로 거절합니다.
- 각 context당 1회 실행합니다. 동일 context/seed/설정을 사용한 작업을 제출해 비교하세요.
- 기존 평가 루프와 모델 로더를 재사용하며 별도 output root에 결과를 생성합니다. 원본 실험 결과·가중치를 덮어쓰지 않습니다.
- 결과에 체크포인트/설정/정규화 및 평가 소스 해시를 기록합니다. 체크포인트 입력이 실행 중 변경되면 실패로 처리합니다.
- 실행 환경의 scene dt와 35 substeps로 실제 제어 간격을 기록합니다.
- 실행 스텝 수가 다르면 동일 seed라도 같은 물리 시간의 노이즈가 같다는 뜻은 아닙니다. 개별 영상의 success는 전체 성공률이 아닙니다.

**로봇 영상** 옵션은 MuJoCo `scene.model/data`를 읽는 선택적 렌더러입니다. Linux CPU headless는 OSMesa 라이브러리(예: `libosmesa6`)와 `MUJOCO_GL=osmesa`, macOS는 GLFW를 사용합니다. macOS의 GPU 미사용 소프트웨어 렌더링까지 보장하지는 않습니다. CPU 추론과 2D 출력은 이 OpenGL 경로를 사용하지 않습니다. 캡처는 step을 추가 진행하지 않으며 카메라는 고정 사선/탑뷰입니다.

macOS CPU에서 baseline 체크포인트의 실제 rollout과 관절 기록은 아래의 2026-10-23 실행으로 확인했습니다. MuJoCo 녹화 카메라 프레이밍, OSMesa 호환성, 렌더링 전후 수치 일치는 별도 검증이 필요합니다. 실패는 작업 로그에 표시됩니다. 서로 다른 컴퓨터에서의 bitwise 재현성을 보장하지 않습니다.

## 검증

```bash
.venv-visualizer/bin/python -m pip install httpx
.venv-visualizer/bin/python -m unittest discover -s tests -p test_visualizer.py -v
```

합성 fixture로 NPZ 입력/비정상 입력 거절, 로컬 origin 제한, 환경 경로 오류, MP4/GIF 합성 크기 및 전체 디코딩을 확인합니다. 합성 fixture는 실제 로봇 성능 결과가 아니며 테스트용 임시 저장소만 사용합니다.

실험 환경을 갖춘 컴퓨터에서는 각 태스크의 짧은 rollout을 캡처 on/off로 실행해 action/state/success 일치를 확인한 뒤 전체 에피소드를 생성하세요. 이후 같은 context의 서로 다른 길이 rollout을 비교해 종료 프레임 유지, 영상 타임라인, 카메라 시야를 확인합니다.

## 파일 없이 보는 3D 장면

첫 화면에는 아래 기본 비교가 자동 표시됩니다. **모두 비우기**를 누르면 Panda 로봇팔의 실제 시각화 메시, 작업대와 블록 예시가 표시됩니다. 드래그로 회전, 휠/핀치로 확대, 우클릭으로 이동하고 사선·탑뷰·정면 버튼으로 시점을 초기화할 수 있습니다. Task 선택에 따른 Pushing / Inserting 예시 배치와 관절 예시 애니메이션은 파일 없이 동작합니다. 이 배치·동작은 학습된 정책이나 실험 결과가 아닌 설명용 미리보기입니다.

브라우저 3D는 Three.js / WebGL로 실행하며 서버에는 GPU나 시뮬레이터가 필요하지 않습니다. JavaScript와 로봇 메시를 저장소에 포함하여 런타임 CDN 접속 없이 동작합니다. WebGL 사용이 불가능하면 오류를 표시하며 기존 2D 보기로 전환할 수 있습니다.

롤아웃을 추가하면 3D 패널에서 기록된 손 좌표와 블록 XY/yaw를 재생합니다. 신규 웹 평가에는 `--record-joints`를 추가하여 post-step 관절 7개를 `<rollout>_joints`에 기록하고 브라우저 Panda 모델의 관절 재생에 사용합니다. 이 옵션을 지정하지 않은 기존 평가 CLI 동작과 기존 NPZ는 유지됩니다. 관절 기록이 없는 과거 파일은 팔을 기본 자세로 두고 손 마커·블록 경로만 재생하며, 없는 팔 자세를 IK로 추정하지 않습니다. 블록 크기/높이와 기본 지면은 표시용입니다. D3IL의 원본 벽/목표/도구 형상을 완전 복제한 장면은 아닙니다.

3D 장면도 현재 카메라와 패널 배치로 GIF/MP4를 출력합니다. 브라우저에서 고정 시간 간격으로 장면을 렌더링하고 로컬 FFmpeg로 인코딩합니다. UI 버튼과 설정창은 영상에 포함하지 않습니다. 관절이 없는 기록은 영상에도 기준 자세임을 표시합니다. 실제 실험 환경의 MuJoCo 녹화 렌더러 정합성 검증은 별도입니다.

### 외부 자산

- Panda 모델: Google DeepMind `mujoco_menagerie/franka_emika_panda`, Apache-2.0. 원본 XML·LICENSE·README 및 정확한 revision/파일 해시는 `static/panda/`에 보존했습니다. `model.json`은 XML의 시각화 body 계층만 추출한 파생 데이터입니다.
- Three.js 0.170.0 / OrbitControls / OBJLoader: MIT, `static/vendor/THREE-LICENSE.txt` 포함.
- 재생성: `python3 visualizer/scripts/vendor_scene.py`. 기존 provenance의 고정 revision을 재사용합니다.

## 2026-10-23 macOS CPU 실제 실행 확인

Apple Silicon 맥에서 `.local-runtime/venv` (Python 3.10.19, PyTorch 2.3.1, MuJoCo 2.3.2, NumPy 1.26.4, Pinocchio 2.7.0)로 두 baseline EMA 체크포인트를 실행했습니다. GPU/MPS는 사용하지 않았습니다. 학습 없이 context 0, seed 0, paired fresh noise, 실행 horizon 8, Euler 10으로 각각 한 에피소드만 생성했습니다.

| 태스크 | 스텝 | 시뮬레이션 길이 | 평가 루프 벽시계 시간 | 개별 결과 |
|---|---:|---:|---:|---|
| Pushing test context 0 | 223 | 7.805초 | 4.11초 | 성공 |
| Inserting val context 0 | 1,249 | 43.715초 | 26.69초 | 성공 |

이는 로컬 단일 rollout 확인이며 전체 성능/성공률 측정이 아닙니다. 서버와의 bitwise 결과 일치는 검증하지 않았습니다. 관절 배열은 각각 `(223, 7)`, `(1249, 7)`이고 유한값과 실제 움직임을 확인했습니다. 브라우저 Panda 모델의 forward kinematics로 계산한 TCP와 기록된 손 위치의 최대 차이는 각각 약 0.393mm, 0.523mm였습니다. 완전한 수치 일치나 전체 물리 장면의 동일성을 뜻하지 않습니다.

로컬 결과 ID는 `8063122c5d304a019fbc12b7f5a61633` (Pushing), `a27e9bd814a24269843aca45bc771262` (Inserting)입니다. `.visualizer/<ID>/`에 NPZ/JSON, checkpoint/source hash, `runtime.json`, `joint_verification.json`, `d3il-local-compatibility.patch`를 저장했습니다. 이 두 CPU 관절 기록은 실제 체크포인트를 실행해 확인한 결과입니다. MuJoCo 녹화 렌더러/OSMesa와 렌더 on/off 일치 검증은 여전히 별도입니다.

로컬 실행 환경 경로:

- 실험 Python: `<저장소>/.local-runtime/venv/bin/python`
- D3IL: `<저장소>/.local-runtime/d3il` (revision `1d9c71850d4fc1477cee17b557c8b97d70b13071`)
- HardFlow: `<저장소>/.local-runtime/HardFlow` (revision `a85105080dd0a28dca76b49ac0414f275de146ed`)
- data: `<저장소>/data/<태스크>_sincos_v1`

Pinocchio 2.6.21은 macOS arm64 wheel이 없어 2.7.0으로 설치했습니다. 사용하지 않는 PyBullet의 소스 빌드도 현재 Apple compiler에서 실패하여, 로컬 D3IL의 `sims/__init__.py`에서 `pybullet` 모듈이 없을 때 해당 backend import만 건너뛰었습니다. 원본 물리 설정·가중치·평가 프로토콜은 변경하지 않았습니다. 실제 설치 버전 목록은 `.local-runtime/installed-packages.txt`에 있습니다. 이 환경과 생성물은 로컬 전용이며 Git에 포함되지 않습니다.

## 기본 비교 대시보드

접속하면 Pushing Plain/OFF와 CAPS/T+S 원본 pilot 평가의 `c000_t0`가 자동으로 나란히 표시됩니다. `visualizer/samples/`에 선택된 실제 기록과 출처 해시를 포함하므로 새 팀원도 업로드나 체크포인트 실행 없이 볼 수 있습니다. 최초 실행 시 `.visualizer/`로 복사하며 기존 작업/대시보드 설정은 덮어쓰지 않습니다. 기본 패널은 `.visualizer/dashboard.json`에서 지정합니다.

- 비교 조건: 각각 100k 학습, training seed 42, train context 0, trajectory 0, eval seed 0, paired fresh noise, 실행 8 / 계획 16 / Euler 10. 한 장면의 선택은 성능 우위를 증명하지 않습니다.
- 원본 40 rollout 집계(10 contexts × 4)는 별도 접이식 표에 표시합니다. 단일 rollout 성공 여부와 전체 성공률을 구분합니다.
- 과거 OFF/CAPS 파일에는 관절이 없습니다. 로컬에서 같은 행동을 replay했으나 손/블록 상태가 원본과 일치하지 않아 재구성한 관절을 합치지 않았습니다. 반투명 팔은 기준 자세이고 움직이는 손·블록이 원본 기록입니다.
- Inserting은 Mac CPU의 baseline 1M 실제 체크포인트 rollout(관절·블록 XYZ·목표 메타데이터 포함)을 표시합니다. 해당 task의 CAPS 파일은 없으므로 대기 패널을 표시합니다. 100k Pushing과 비교하지 않습니다.

목표는 G1/G2/G3 표식과 연결선, 좌표, 블록별 현재 거리로 표시합니다. Pushing은 서로 다른 목표에 도착하면 색상 교차를 허용하며 50 mm 이내입니다. Inserting은 각 블록의 지정 목표에 10 mm 이내입니다. 목표 메타데이터는 신규 `--record-joints` 평가에서 시뮬레이터로부터 기록합니다. 기존 명시적 task 평가만 검증된 D3IL revision의 목표 정의를 사용합니다.

3D 판정에는 블록 XYZ가 필요합니다. 원본 pilot은 XY/yaw만 있어 **XY 투영 거리**로 표시하며 성공 판정과 동일하다고 주장하지 않습니다. Pushing 평균 거리는 원본 evaluator와 같이 각 블록에서 가까운 목표까지 거리의 평균입니다. 블록별 행은 서로 다른 목표에 대한 최대 거리가 최소가 되도록 짝을 정하므로 이 평균과 다를 수 있습니다. 성공 여부는 원본 결과 값을 표시합니다.

그래프는 목표 평균 거리, 손 추종 오차(XY), 행동 변화량, 행동 크기, 손 속도(XY), 관절 속도 노름을 제공합니다. 시간축은 초, 거리는 mm, 속도는 m/s 또는 rad/s입니다. 행동은 XY 위치 증분이고 변화량은 jerk가 아닙니다. 최초 전이와 누락된 기록은 0으로 채우지 않습니다. 그래프를 클릭하면 모든 장면과 수치가 해당 시점으로 이동하며, 짧은 기록의 그래프는 종료 이후 연장하지 않습니다.

## 밝은 화면과 발표용 출력

UI, 3D 작업대, 그래프, 2D 및 3D 출력은 흰색 배경을 사용합니다. 비교선은 파랑·주황, 손 위치는 파랑, 명령 위치는 회색입니다. 형식/구간/배속을 선택한 뒤 **현재 배치로 내보내기**를 누르고 **작업과 결과 → 영상 다운로드**로 저장합니다.

MP4는 H.264 / yuv420p / BT.709 / faststart, GIF는 15fps입니다. 3D MP4는 패널당 640×480, GIF는 패널당 480×360이며 배치에 따라 합쳐집니다. 3D MP4 출력 길이는 최대 60초, GIF는 최대 15초입니다. 브라우저를 열린 상태로 유지해야 프레임을 생성할 수 있습니다. 캡처 프레임은 총 256 MiB로 제한하고 인코딩 성공 후 제거합니다. 서버 재시작 중 캡처하던 작업은 interrupted로 남습니다. 기존 MuJoCo 녹화 프레임의 내부 배경은 녹화 당시 이미지 그대로이며 흰색 여백에 배치합니다.
