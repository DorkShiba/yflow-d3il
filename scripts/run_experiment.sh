#!/usr/bin/env bash
# 본실험(ID·OOD): 원하는 조합(하나 또는 여러 개)을 맡은 context에 대해 순서대로 실행한다.
# 각 단위(task × 조합 × context, noise seed 0·1·2)는 끝나는 즉시 저장되며, 다시 실행하면 완료된 단위를 건너뛴다.
# 같은 장비에서 여러 프로세스(GPU별 등)를 동시에 띄워도 단위마다 잠금을 잡으므로 같은 단위를 두 번 돌리지 않는다.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"

python_bin="${PYTHON:-python}"
name="" contexts="" ood="" tasks="both" combos="all" device="cuda:0" max_steps="" dry_run=false allow_dirty=false

usage() {
    cat <<'HELP'
Usage: bash scripts/run_experiment.sh --name NAME --contexts LIST [options]

  --name NAME          팀원 이름(영문, 예: hyeongbin). 결과와 docs/report_NAME.md에 기록된다
  --contexts LIST      context 번호(0..29). 예: 0-6, 7-16, 3,5,9. 분담은 README.md "실험 분담" 참고
  --ood COND[,COND]    OOD 조건(예: A_mid, 또는 A_weak,A_strong처럼 여러 개). 생략하면 ID.
                       조건마다 순서대로 실행하고 보고서는 docs/report_ood_COND_NAME.md
  --task T             pushing | inserting | both (기본 both: pushing 먼저)
  --combos LIST        all(기본) | 0-31 | 5,17 | P05,P17. 조합 번호 = ①16+②8+③4+④2+⑤1
  --device DEV         cuda:0(기본) | cuda:1 | cpu. GPU를 요청했는데 쓸 수 없으면 시작하지 않는다
  --smoke N            동작 확인: 최대 N step만 실행하고 결과는 로컬(fm_constraint/runs)에만 저장
  --dry-run            검사만 하고 실행할 단위 목록을 출력
  --allow-dirty        커밋하지 않은 코드 변경이 있어도 실행 (본실험에서는 쓰지 않는다)

예) bash scripts/run_experiment.sh --name hyeongbin --contexts 0-6
    bash scripts/run_experiment.sh --name junhyuk --contexts 7-16 --task inserting --combos 16-31 --device cuda:1
    bash scripts/run_experiment.sh --name hyeongbin --contexts 0-6 --ood A_mid
    bash scripts/run_experiment.sh --name hyeongbin --contexts 0-7 --ood A_weak,A_strong
HELP
}

while (($#)); do
    case "$1" in
        --name) name="$2"; shift 2 ;;
        --contexts) contexts="$2"; shift 2 ;;
        --ood) ood="$2"; shift 2 ;;
        --task) tasks="$2"; shift 2 ;;
        --combos) combos="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --smoke) max_steps="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --allow-dirty) allow_dirty=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$name" && -n "$contexts" ]] || { usage >&2; exit 2; }
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--name must be ASCII letters/digits/_/-" >&2; exit 2; }
[[ "$name" != results && "$name" != ood* ]] || { echo "--name '$name' is reserved" >&2; exit 2; }
# 조건 목록: 빈 문자열 = ID. 'A_weak,A_strong'처럼 쉼표로 여러 개를 주면 차례로 실행한다.
if [[ -n "$ood" ]]; then
    IFS=',' read -r -a conds <<< "$ood"
    for c in "${conds[@]}"; do
        [[ "$c" =~ ^[A-Za-z0-9_]+$ ]] || { echo "bad --ood condition: '$c'" >&2; exit 2; }
    done
    # 실행 전에 모든 조건 이름을 검사한다(오타로 앞 조건만 돌고 멈추지 않게).
    "$python_bin" - "${conds[@]}" <<'PY' || exit 2
import sys
from fm_constraint.experiment import check_ood
for c in sys.argv[1:]:
    check_ood(c)
PY
else
    conds=("")
fi
case "$tasks" in both) tasks=(pushing inserting) ;; pushing|inserting) tasks=("$tasks") ;;
    *) echo "--task must be pushing, inserting or both" >&2; exit 2 ;; esac

# 1. 코드가 커밋된 상태인지 (결과·보고서 파일은 제외)
dirty="$(git status --porcelain -- . ':!experiments' ':!docs' ':!fm_constraint/runs' | grep -v '^??' || true)"
if [[ -n "$dirty" ]] && ! $allow_dirty; then
    echo "커밋하지 않은 코드 변경이 있습니다. 모든 팀원이 같은 코드로 실행해야 합니다:" >&2
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
[[ "$device" == cpu ]] && echo "경고: CPU로 실행합니다. 같은 context의 32조합은 같은 장비에서 돌려야 합니다." >&2

export MUJOCO_GL="${MUJOCO_GL:-disable}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
log_dir="fm_constraint/runs/experiment_logs"
mkdir -p "$log_dir"

report_files=()
for cond in "${conds[@]}"; do
    ood_args=()
    if [[ -n "$cond" ]]; then
        ood_args=(--ood "$cond")
        report_files+=("docs/report_ood_${cond}_${name}.md")
    else
        report_files+=("docs/report_${name}.md")
    fi
    for task in "${tasks[@]}"; do
        command=("$python_bin" -u -m fm_constraint.experiment run --task "$task" --combos "$combos"
                 --contexts "$contexts" --device "$device" --name "$name" "${ood_args[@]}")
        [[ -n "$max_steps" ]] && command+=(--max-steps "$max_steps")
        if $dry_run; then
            printf '%q ' "${command[@]}"; printf '\n'
            continue
        fi
        # 같은 이름으로 여러 GPU·범위를 동시에 실행해도 로그가 겹치지 않게 장치·범위·PID를 넣는다.
        tag="${name}${cond:+_ood_$cond}_${task}_${device//:/}_ctx${contexts//[,]/_}_c${combos//[,]/_}"
        log="$log_dir/${tag}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
        echo "== ${cond:-id} $task: combos $combos, contexts $contexts → $log"
        "${command[@]}" 2>&1 | tee "$log"
    done
    # 조건 하나가 끝날 때마다 보고서를 만든다(다음 조건 도중 중단돼도 끝난 조건은 공유할 수 있다).
    if ! $dry_run && [[ -z "$max_steps" ]]; then
        "$python_bin" -m fm_constraint.experiment report --name "$name" "${ood_args[@]}"
    fi
done

if ! $dry_run && [[ -z "$max_steps" ]]; then
    echo
    echo "완료. 결과(experiments/)는 로컬에만 남고, 보고서만 공유한다:"
    echo "  git add ${report_files[*]} && git commit -m \"report: ${name}${ood:+ ${ood}} contexts ${contexts}\" && git pull --rebase && git push"
fi
