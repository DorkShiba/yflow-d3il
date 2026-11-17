#!/usr/bin/env bash
# 추가 OOD A 강도 실험(2026-11-17): Pushing·Inserting A_10(10 cm), Stacking A_3(3 cm). 한 사람이 한 태스크의
# 30 context × 32조합 전체를 맡는다(준혁 GPU 사용 불가로 context 분담 대신 태스크 분담).
#
#   hyeongbin  Inserting A_10  → docs/report_ood_A_10_hyeongbin.md
#   jaeyeop    Pushing   A_10  → docs/report_ood_A_10_jaeyeop.md
#   seongmin   Stacking  A_3   → docs/stacking/report_A_3_seongmin.md
#
# 단위가 많아(960개) 같은 GPU에 worker 여러 개를 띄운다. 모든 worker가 같은 단위 목록을 같은 순서로 돌며,
# 단위마다 잠금(flock)을 잡고 결과가 있으면 건너뛰므로 같은 단위를 두 번 돌리지 않는다.
# 중단돼도 같은 명령을 다시 실행하면 끝난 단위를 건너뛰고 이어서 실행한다. 보고서는 모든 worker가 끝난 뒤 한 번 만든다.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"

python_bin="${PYTHON:-python}"
member="" contexts="0-29" combos="all" device="cuda:0" workers=4 max_steps="" dry_run=false allow_dirty=false

usage() {
    cat <<'HELP'
Usage: bash scripts/run_ood_extra.sh --member NAME [options]

  --member NAME      hyeongbin (Inserting A_10) | jaeyeop (Pushing A_10) | seongmin (Stacking A_3)
  --workers N        같은 장치에 띄울 프로세스 수(기본 4). 시뮬레이터가 CPU를 쓰므로 CPU 코어 수 이하로
  --device DEV       cuda:0(기본) | cuda:1 | cpu
  --contexts LIST    context 번호(기본 0-29, 전체). 동작 확인이나 나눠 돌릴 때만 바꾼다
  --combos LIST      all(기본) | 0-31 | 0,29
  --smoke N          동작 확인: 최대 N step, 결과는 로컬(fm_constraint/runs)에만, 보고서 없음
  --dry-run          검사만 하고 실행할 명령을 출력
  --allow-dirty      커밋하지 않은 코드 변경이 있어도 실행 (본실험에서는 쓰지 않는다)

예) bash scripts/run_ood_extra.sh --member hyeongbin --smoke 40 --contexts 0 --combos 0,31 --workers 1
    bash scripts/run_ood_extra.sh --member hyeongbin --workers 6
HELP
}

while (($#)); do
    case "$1" in
        --member) member="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --contexts) contexts="$2"; shift 2 ;;
        --combos) combos="$2"; shift 2 ;;
        --smoke) max_steps="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --allow-dirty) allow_dirty=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "--workers must be a positive integer" >&2; exit 2; }

# 분담표: 이름 → (runner, task, 조건)
case "$member" in
    hyeongbin) runner=main task=inserting cond=A_10 ;;
    jaeyeop)   runner=main task=pushing   cond=A_10 ;;
    seongmin)  runner=stacking task=stacking cond=A_3 ;;
    *) echo "--member must be hyeongbin, jaeyeop or seongmin" >&2; usage >&2; exit 2 ;;
esac

if [[ "$runner" == main ]]; then
    run_cmd=("$python_bin" -u -m fm_constraint.experiment run --task "$task" --ood "$cond" --combos "$combos"
             --contexts "$contexts" --device "$device" --name "$member")
    report_cmd=("$python_bin" -m fm_constraint.experiment report --name "$member" --ood "$cond")
    report_file="docs/report_ood_${cond}_${member}.md"
else
    run_cmd=("$python_bin" -u -m fm_constraint.stacking_experiment run --cond "$cond" --combos "$combos"
             --contexts "$contexts" --device "$device" --name "$member")
    $allow_dirty && run_cmd+=(--allow-dirty)
    report_cmd=("$python_bin" -m fm_constraint.stacking_experiment report --cond "$cond" --name "$member")
    report_file="docs/stacking/report_${cond}_${member}.md"
fi
[[ -n "$max_steps" ]] && run_cmd+=(--max-steps "$max_steps")

# 0. 조건·데이터 검사: 조건 이름, OOD 파일이 manifest 해시와 맞는지(데이터를 pull하지 않았으면 여기서 멈춘다)
"$python_bin" - "$runner" "$task" "$cond" <<'PY' || exit 2
import sys
runner, task, cond = sys.argv[1:]
if runner == 'main':
    from configs.config import REPO
    from fm_constraint.experiment import check_ood
    from ood.apply import load_cases
    check_ood(cond)
    cases, info = load_cases(REPO / 'data/ood', task, cond)
    assert len(cases) > 0, f'no {cond} rows for {task}'
    print(f'OOD data OK: {task} {cond} {len(cases)} rows, sha256 {info["sha256"][:12]}')
else:
    from fm_constraint.stacking_experiment import check_cond, condition_contexts, load_config
    check_cond(cond)
    ctxs, meta = condition_contexts(load_config(), cond)
    print(f'OOD data OK: stacking {cond} {len(ctxs)} contexts, level {meta["level_m"]} m, sha256 {meta["sha256"][:12]}')
PY

# 1. 코드가 커밋된 상태인지 (결과·보고서 파일은 제외)
dirty="$(git status --porcelain -- . ':!experiments' ':!docs' ':!fm_constraint/runs' | grep -v '^??' || true)"
if [[ -n "$dirty" ]] && ! $allow_dirty; then
    echo "커밋하지 않은 코드 변경이 있습니다. git pull 후 변경 없이 실행하세요:" >&2
    echo "$dirty" >&2
    exit 1
fi
echo "commit $(git rev-parse --short HEAD)"

# 2. GPU 검사: 요청한 장치에서 실제로 계산이 되는지 확인한다. CPU로 몰래 넘어가지 않는다.
if [[ "$device" == cuda* ]]; then
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi가 없습니다. GPU 드라이버를 확인하세요" >&2; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
fi
"$python_bin" - "$device" <<'PY'
import sys
from fm_constraint.experiment import device_info
info = device_info(sys.argv[1])
print('device OK:', info.get('gpu', 'cpu'), '| torch', info['torch'], '| cuda', info.get('cuda'))
PY
[[ "$device" == cpu ]] && echo "경고: CPU로 실행합니다. 32조합은 모두 같은 장비에서 돌려야 합니다." >&2
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
tag="${member}_${task}_${cond}_${device//:/}_ctx${contexts//[,]/_}_c${combos//[,]/_}"

pids=()
trap 'echo "중단: worker 종료 중" >&2; kill "${pids[@]}" 2>/dev/null || true; wait || true; exit 130' INT TERM
for ((w = 1; w <= workers; w++)); do
    log="$log_dir/${tag}_${stamp}_w${w}_$$.log"
    echo "== worker $w: $task $cond, combos $combos, contexts $contexts → $log"
    "${run_cmd[@]}" >"$log" 2>&1 &
    pids+=("$!")
    sleep 2   # 모델 로딩이 한꺼번에 몰리지 않게
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

[[ -n "$max_steps" ]] && { echo "smoke 완료(보고서 없음)."; exit 0; }
"${report_cmd[@]}"
echo
echo "완료. 결과(experiments/)는 로컬에만 남고, 보고서만 공유한다:"
echo "  git add $report_file && git commit -m \"report: $member $task $cond contexts $contexts\" && git pull --rebase && git push"
