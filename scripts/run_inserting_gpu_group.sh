#!/usr/bin/env bash
# 동일 manifest로 Inserting을 두 그룹(4조합씩)으로 나누어 실행한다.
# 기존 runner를 수정하지 않아 동결된 실험의 코드 hash를 유지한다.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"

python_bin="${PYTHON:-python}"
group=""
run_root="fm_constraint/runs/inserting_gpu_main"
stage="main"
device="cuda:0"
inserting_set="holdout"
split_seed=2026
dry_run=false
extra=()

usage() {
    cat <<'HELP'
Usage: bash scripts/run_inserting_gpu_group.sh --group 1|2 [options]
  --group 1             I00 I04 I08 I12
  --group 2             I16 I20 I24 I28
  --output PATH         두 그룹 모두 같은 경로 지정. 기본 fm_constraint/runs/inserting_gpu_main
  --device DEVICE       기본 cuda:0. 각 Slurm GPU 할당 셸에서는 둘 다 cuda:0 사용
  --stage STAGE         기본 main; 검증용 smoke / pilot도 지원
  --inserting-set SET   기본 holdout; 기존 실행이 all-val이면 동일하게 지정
  --split-seed N        기본 2026; 기존 실행과 동일하게 지정
  --dry-run            목록·설정 검증 후 4개 명령만 출력. GPU/시뮬레이터 실행 안 함
  --context-limit N     smoke에서만 context 수 축소
  --max-steps N         smoke에서만 step 제한 변경
  --help

GPU 하나씩 할당받은 두 셸에서 --group 1 / --group 2를 각각 실행한다.
단일 셸에 GPU 두 개를 모두 할당받았다면 CUDA_VISIBLE_DEVICES=0 / 1로
각 프로세스에 하나씩 노출하고, 두 스크립트의 논리 장치는 모두 cuda:0으로 둔다.
기존 8개 순차 실행을 먼저 중단해야 한다. 이 스크립트는 다른 프로세스를 종료하지 않는다.
완료 조합은 건너뛰며, 중단된 조합은 원본을 보존하고 전체 context×seed를 재실행한다.
HELP
}

while (($#)); do
    case "$1" in
        --group|--output|--device|--stage|--inserting-set|--split-seed|--context-limit|--max-steps)
            if (($# < 2)); then echo "Missing value: $1" >&2; exit 2; fi
            case "$1" in
                --group) group="$2" ;;
                --output) run_root="$2" ;;
                --device) device="$2" ;;
                --stage) stage="$2" ;;
                --inserting-set) inserting_set="$2" ;;
                --split-seed) split_seed="$2" ;;
                *) extra+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$group" in
    1) jobs=(I00 I04 I08 I12) ;;
    2) jobs=(I16 I20 I24 I28) ;;
    *) echo '--group must be 1 or 2' >&2; usage >&2; exit 2 ;;
esac

export MUJOCO_GL="${MUJOCO_GL:-disable}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/yflow-inserting-group-${group}-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Python에서도 입력 보호 및 출력 경로를 확인한다.
run_root="$("$python_bin" -c 'import sys; from configs.config import ExperimentConfig, REPO; cfg=ExperimentConfig.from_yaml(REPO/"configs/pushing.yaml"); print(cfg.require_output(sys.argv[1]))' "$run_root")"
mkdir -p -- "$run_root"

# 기존 16개 전체 스크립트의 exclusive lock과 충돌하며, 서로 다른 그룹은 병렬 허용한다.
exec 8>"$run_root/.suite.lock"
if ! flock -s -n 8; then
    echo "Whole-suite runner is using $run_root; stop it first." >&2
    exit 1
fi
exec 9>"$run_root/.inserting_group_${group}.lock"
if ! flock -n 9; then
    echo "Inserting group $group is already running in $run_root" >&2
    exit 1
fi

if ! "$dry_run"; then
    "$python_bin" - "$device" <<'PY' | tee -a "$run_root/group_${group}.device.log"
import os
import socket
import sys
import torch
print("Host:", socket.gethostname())
print("Slurm job:", os.environ.get("SLURM_JOB_ID"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
name = sys.argv[1]
if name.startswith('cuda'):
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: check the GPU allocation and active Python environment')
    device = torch.device(name)
    print('Python:', sys.executable)
    print('Logical device:', device)
    print('GPU:', torch.cuda.get_device_name(device))
else:
    print('Device:', name)
PY
fi

# 동시 최초 실행에서도 manifest/config 생성은 한 프로세스만 수행한다.
(
    flock -x 7
    "$python_bin" -u -m fm_constraint.method123 prepare \
        --output "$run_root" --stage "$stage" --device "$device" \
        --inserting-set "$inserting_set" --split-seed "$split_seed" "${extra[@]}"
) 7>"$run_root/.prepare.lock"

for index in "${!jobs[@]}"; do
    job="${jobs[$index]}"
    command=("$python_bin" -u -m fm_constraint.method123 run --output "$run_root" --job "$job")
    if "$dry_run"; then
        printf '[group %s: %d/4] ' "$group" "$((index + 1))"
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        printf '[group %s: %d/4] %s — RTC OFF / LiPo OFF\n' "$group" "$((index + 1))" "$job"
        "${command[@]}" 2>&1 | tee -a "$run_root/${job}.console.log"
    fi
done
printf 'Group %s finished. Output: %s\n' "$group" "$run_root"
# 기존 summarize는 16개 조합 전체를 요구하므로 여기서는 자동 호출하지 않는다.
