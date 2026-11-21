# Inserting · A_10 학습 밀도 영상

분석 PPT 3–4장의 로봇팔 비교 영상을 생성한다. 배치 13(`env_050_05.npz`)에서 세 궤적을 MuJoCo로 재생하고 영상 두 편을 저장한다.

| 영상 | 왼쪽 | 오른쪽 |
|---|---|---|
| `A10-배치13-PF2MP-학습밀도.mp4` | 기본 정책, seed 0 | ② PF2MP, seed 0 |
| `A10-배치13-PF2MP-성공실패-밀도.mp4` | ② PF2MP, seed 0 | ② PF2MP, seed 1 |

## 실행

D3IL Inserting 평가가 가능한 Python 환경, GLFW용 디스플레이, FFmpeg(libx264), Pillow, 한글 글꼴이 필요하다. 발표에 사용한 조합은 이 저장소의 평가 코드 `8d0c7aba`, D3IL `1d9c718`, macOS CPU다. 아래 명령은 영상 코드가 있는 저장소 루트에서 실행한다.

```sh
git worktree add --detach ../yflow-density-eval 8d0c7aba
python scripts/inserting_density_video/generate.py \
  --source ../yflow-density-eval \
  --d3il /path/to/d3il \
  --output ./outputs/inserting-density-videos
```

기본 실행은 세 정책·seed 조합을 평가하고 영상까지 생성한다. 평가 체크아웃에 Inserting 학습 데이터, 기본 정책 체크포인트, A_10 배치 파일이 필요하다. ②의 첫 평가가 학습 데이터에서 PF2MP 캐시를 준비한다. Linux에서는 `--font /path/to/NotoSansCJK-Regular.ttc`처럼 한글 글꼴을 지정하고, 화면이 없는 환경에서는 `xvfb-run -a`로 실행한다.

기존 CPU 실행을 재생하려면 위 명령에 `--runs /path/to/A_10/env_050_05.npz`를 추가한다. 이 경로는 디렉터리이며 `I00-seed0/`, `I08-seed0/`, `I08-seed1/` 각각에 단일 에피소드의 `rollouts.json`과 `results.json`이 있어야 한다. 이 방식은 같은 학습 데이터로 만든 PF2MP 캐시를 사용한다. 렌더러는 매 스텝의 손·블록 위치를 기록과 대조한다.

## 영상 읽는 법

- 바닥색은 시연에서 손 목표 XY의 KDE 밀도가 5·50·75백분위 경계 중 어디에 있는지 나타낸다. 대역폭은 2.5 cm이며, 두 영상의 모든 패널에 동일한 학습 분포를 적용한다. 로봇팔과 블록은 바닥색을 가린다.
- 체류율은 각 스텝의 손 목표 XY가 학습 밀도 75백분위를 초과한 누적 비율이다. 두 실행 중 하나가 끝나면 체류율의 분모를 고정하고 나머지 실행은 끝까지 보여준다.
- 블록별로 목표 XY에서 1 cm 이내에 처음 들어온 시점부터 도달 개수를 누적한다. 발표 재실행에서 첫 1,800스텝의 고밀도 체류는 기본 seed 0이 76스텝, ② seed 0이 718스텝, ② seed 1이 0스텝이다. 최종 도달 개수는 각각 1·3·0개다.
