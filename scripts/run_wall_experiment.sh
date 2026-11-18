#!/usr/bin/env bash
# 벽 OOD 본실험(2026-11-18): Pushing·Inserting 장면에 학습 때 없던 벽, 32조합 × 30 context × noise 3.
# 벽 회피 부등식은 ①·③(⑤는 ①을 따라감)에만 들어간다. 정의는 fm_constraint/wall.py.
#
# 같은 GPU에 worker 여러 개를 띄운다. 모든 worker가 같은 단위 목록을 같은 순서로 돌며 단위마다 잠금을 잡고,
# 결과가 있으면 건너뛰므로 같은 단위를 두 번 돌리지 않는다. 중단 후 같은 명령을 다시 실행하면 이어서 한다.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"

python_bin="${PYTHON:-python}"
name="" task="" contexts="0-29" combos="all" device="cuda:0" workers=4 smoke="" dry_run=false allow_dirty=false

usage() {
    cat <<'HELP'
Usage: bash scripts/run_wall_experiment.sh --name NAME --task pushing|inserting [options]

  --name NAME        팀원 이름(영문). docs/report_wall_<NAME>.md
  --task TASK        pushing | inserting
  --workers N        같은 장치에 띄울 프로세스 수(기본 4). CPU 코어 수 이하로
  --device DEV       cuda:0(기본) | cuda:1 | cpu
  --contexts LIST    context 번호(기본 0-29)
  --combos LIST      all(기본) | 0-31 | 0,16
  --smoke N          동작 확인: 최대 N step, 결과는 로컬(fm_constraint/runs)에만, 보고서 없음
  --dry-run          검사만 하고 실행할 명령을 출력
  --allow-dirty      커밋하지 않은 코드 변경이 있어도 실행 (본실험에서는 쓰지 않는다)

예) bash scripts/run_wall_experiment.sh --name hyeongbin --task pushing --smoke 40 --contexts 0 --combos 0,16 --workers 1
    bash scripts/run_wall_experiment.sh --name hyeongbin --task pushing --workers 6
HELP
}

while (($#)); do
    case "$1" in
        --name) name="$2"; shift 2 ;;
        --task) task="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --contexts) contexts="$2"; shift 2 ;;
        --combos) combos="$2"; shift 2 ;;
        --smoke) smoke="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --allow-dirty) allow_dirty=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$name" ]] || { echo "--name is required" >&2; exit 2; }
[[ "$task" == pushing || "$task" == inserting ]] || { echo "--task must be pushing or inserting" >&2; exit 2; }
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "--workers must be a positive integer" >&2; exit 2; }

run_cmd=("$python_bin" -u -m fm_constraint.wall_experiment run --task "$task" --combos "$combos"
         --contexts "$contexts" --device "$device" --name "$name")
[[ -n "$smoke" ]] && run_cmd+=(--smoke "$smoke")
report_cmd=("$python_bin" -m fm_constraint.wall_experiment report --name "$name")
report_file="docs/report_wall_${name}.md"

# 1. 코드가 커밋된 상태인지 (결과·보고서 파일은 제외)
dirty="$(git status --porcelain -- . ':!experiments' ':!docs' ':!fm_constraint/runs' | grep -v '^??' || true)"
if [[ -n "$dirty" ]] && ! $allow_dirty; then
    echo "커밋하지 않은 코드 변경이 있습니다. git pull 후 변경 없이 실행하세요:" >&2
    echo "$dirty" >&2
    exit 1
fi
echo "commit $(git rev-parse --short HEAD)"

# 2. GPU 검사
"$python_bin" - "$device" <<'PY'
import sys
from fm_constraint.experiment import device_info
info = device_info(sys.argv[1])
print('device OK:', info.get('gpu', 'cpu'), '| torch', info['torch'], '| cuda', info.get('cuda'))
PY
cores="$(nproc 2>/dev/null || echo 0)"
((cores > 0 && workers > cores)) && echo "경고: worker $workers개 > CPU 코어 $cores개" >&2

if $dry_run; then
    echo "workers: $workers"
    printf '%q ' "${run_cmd[@]}"; printf '\n'
    printf '%q ' "${report_cmd[@]}"; printf '\n'
    exit 0
fi

export MUJOCO_GL="${MUJOCO_GL:-disable}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
log_dir="fm_constraint/runs/experiment_logs"
mkdir -p "$log_dir"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
tag="${name}_${task}_wall_${device//:/}_ctx${contexts//[,]/_}_c${combos//[,]/_}"

pids=()
trap 'echo "중단: worker 종료 중" >&2; kill "${pids[@]}" 2>/dev/null || true; wait || true; exit 130' INT TERM
for ((w = 1; w <= workers; w++)); do
    log="$log_dir/${tag}_${stamp}_w${w}_$$.log"
    echo "== worker $w: $task wall, combos $combos, contexts $contexts → $log"
    "${run_cmd[@]}" >"$log" 2>&1 &
    pids+=("$!")
    sleep 2
done
echo "진행 확인: tail -f $log_dir/${tag}_${stamp}_w*_$$.log"

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done
trap - INT TERM
if ((failed)); then
    echo "worker $failed개가 오류로 끝났습니다. 로그를 확인한 뒤 같은 명령으로 다시 실행하면 이어서 진행합니다." >&2
    exit 1
fi

[[ -n "$smoke" ]] && { echo "smoke 완료(보고서 없음)."; exit 0; }
"${report_cmd[@]}"
"$python_bin" -m fm_constraint.wall_experiment summary --task "$task" --out "docs/wall_summary_${task}_${name}.md"
echo
echo "완료. 결과(experiments/wall_v1)는 로컬에만 남고, 보고서만 공유한다:"
echo "  git add $report_file docs/wall_summary_${task}_${name}.md && git commit -m \"report: $name $task wall\" && git pull --rebase && git push"
