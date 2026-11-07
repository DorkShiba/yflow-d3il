#!/usr/bin/env bash
# 5개 방법(① Projection ② PF2MP ③ Filter ④ RTC ⑤ LiPo)의 태스크당 32개 조합을 순차 실행한다.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"

python_bin="${PYTHON:-python}"
stage="main"
device="${DEVICE:-cpu}"
inserting_set="holdout"
split_seed=2026
dry_run=false
run_root=""
extra=()

usage() {
    cat <<'HELP'
Usage: bash scripts/run_suite.sh [options]
  --stage smoke|pilot|main     기본 main (태스크별 60 context × seed 0..4)
  --device cpu|cuda:0          기본 cpu; DEVICE 환경변수 사용 가능
  --inserting-set holdout|all-val
                              기본 holdout: 20 tuning group / 60 final group
  --output PATH               fm_constraint/runs/ 아래 저장; 같은 경로로 재실행하면 완료 조합 건너뜀
  --split-seed N              Inserting 그룹 분할 seed, 기본 2026
  --dry-run                   목록·설정만 저장하고 64개 실행 명령 출력 (시뮬레이터 실행 안 함)
  --context-limit N           smoke에서만 context 수 축소
  --max-steps N               smoke에서만 제어 step 수 변경 (기본 10)
  --help

PYTHON=/path/to/python 으로 PyTorch·D3IL이 설치된 interpreter를 지정할 수 있다.
순서: P00..P31 → I00..I31. 비트 ①=16 ②=8 ③=4 ④=2 ⑤=1.
③은 실행 직전 H=1 필터, ⑤는 ①이 켜져 있으면 ① 제약을 포함한 jerk QP다.
HELP
}

while (($#)); do
    case "$1" in
        --stage|--device|--inserting-set|--output|--split-seed|--max-steps|--context-limit)
            if (($# < 2)); then echo "Missing value: $1" >&2; exit 2; fi
            case "$1" in
                --stage) stage="$2" ;;
                --device) device="$2" ;;
                --inserting-set) inserting_set="$2" ;;
                --output) run_root="$2" ;;
                --split-seed) split_seed="$2" ;;
                *) extra+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$run_root" ]]; then
    run_root="$repo_dir/fm_constraint/runs/suite_${stage}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
# Python에서도 출력 경로를 검증하며 기존 완료 결과를 덮어쓰지 않는다.
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/yflow-suite-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$python_bin" -u -m fm_constraint.suite prepare \
    --output "$run_root" --stage "$stage" --device "$device" \
    --inserting-set "$inserting_set" --split-seed "$split_seed" "${extra[@]}"
# 루트 경로는 repo 기준 절대 경로로 확정한다.
run_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$run_root")"
exec 9>"$run_root/.suite.lock"
if ! flock -n 9; then
    echo "Another process is using $run_root" >&2
    exit 1
fi

jobs=($(printf 'P%02d ' $(seq 0 31)) $(printf 'I%02d ' $(seq 0 31)))
for index in "${!jobs[@]}"; do
    job="${jobs[$index]}"
    command=("$python_bin" -u -m fm_constraint.suite run --output "$run_root" --job "$job")
    if "$dry_run"; then
        printf '[%02d/%d] ' "$((index + 1))" "${#jobs[@]}"
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        printf '[%02d/%d] %s\n' "$((index + 1))" "${#jobs[@]}" "$job"
        "${command[@]}" 2>&1 | tee -a "$run_root/${job}.console.log"
    fi
done
if ! "$dry_run"; then
    "$python_bin" -u -m fm_constraint.suite summarize --output "$run_root" \
        2>&1 | tee -a "$run_root/summary.console.log"
fi
printf 'Output: %s\n' "$run_root"
