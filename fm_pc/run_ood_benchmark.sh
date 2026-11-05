#!/usr/bin/env bash
# OOD benchmark: ID + OOD-1..4 for one task and one method, then ood_metrics.py.
#
#   bash fm_pc/run_ood_benchmark.sh pushing                       # baseline policy
#   bash fm_pc/run_ood_benchmark.sh pushing --sampler pc --constraints action_box,tracking
#   TAG=pc QUICK=1 bash fm_pc/run_ood_benchmark.sh inserting --sampler pc ...
#   RUN=fm_baseline/results/stacking_main bash fm_pc/run_ood_benchmark.sh stacking
#
# Tasks: pushing inserting avoiding aligning sorting_2 sorting_4 sorting_6 stacking.
# Inserting uses eval_inserting.py (its own context file); every other task goes
# through the generic eval_task.py (identical to eval_pushing.py for Pushing).
# Extra arguments are passed to the evaluator (method flags).  Env vars:
#   RUN       trained run dir (default: fm_baseline/results/<task>_baseline)
#   TAG       prefix of every run tag (default: base)
#   QUICK=1   small smoke run (4 contexts x 1)
#   JOBS      evaluations in parallel (default 1)
#   CONDS     space-separated subset of condition names below (default: all)
#   SEED      policy seed (default 0); the OOD draws use --ood-seed 0
set -euo pipefail
TASK=${1:?usage: run_ood_benchmark.sh <task> [method flags...]}
shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN=${RUN:-$REPO/fm_baseline/results/${TASK}_baseline}
TAG=${TAG:-base}
JOBS=${JOBS:-1}
SEED=${SEED:-0}

# contexts x rollouts per condition (full run).  D3IL's own protocol in brackets.
case $TASK in
  pushing)   SIZE=(--n-contexts 30 --n-traj 4) ;;      # [D3IL: 60 test contexts]
  inserting) SIZE=(--n-traj 4) ;;                       # all validation contexts
  avoiding)  SIZE=(--n-contexts 100 --n-traj 1) ;;     # [100 rollouts, fixed start]
  aligning)  SIZE=(--n-contexts 60 --n-traj 4) ;;      # [60 x 8]
  sorting_2|sorting_4|sorting_6) SIZE=(--n-contexts 30 --n-traj 4) ;;   # [30 x 1 / 12 / 12]
  stacking)  SIZE=(--n-contexts 30 --n-traj 4) ;;      # [5 x 1 -- too few for CIs]
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
[[ ${QUICK:-0} == 1 ]] && SIZE=(--n-contexts 4 --n-traj 1)
if [[ $TASK == inserting ]]; then
  EVAL=(python "$REPO/fm_pc/eval_inserting.py")
else
  EVAL=(python "$REPO/fm_pc/eval_task.py" --task "$TASK")
fi
EVAL+=(--run-dir "$RUN" --ckpt "${CKPT:-baseline_ema.pt}" --execute-steps 8 --seed "$SEED" "${SIZE[@]}")

# name | OOD flags   (same magnitudes for every task; edit here, not in the code)
declare -A COND=(
  [id]="--ood none"
  [shift2cm]="--ood shift --ood-shift 0.02"
  [shift4cm]="--ood shift --ood-shift 0.04"
  [noise5mm]="--ood obsnoise --ood-noise-pos 0.005 --ood-noise-yaw 5"
  [noise10mm]="--ood obsnoise --ood-noise-pos 0.010 --ood-noise-yaw 5"
  [push3cm]="--ood push --ood-push-dist 0.03"
  [push5cm]="--ood push --ood-push-dist 0.05"
  [delay1]="--ood delay --ood-delay 1"
  [delay2]="--ood delay --ood-delay 2"
  [delay4]="--ood delay --ood-delay 4"
)
ORDER=(id shift2cm shift4cm noise5mm noise10mm push3cm push5cm delay1 delay2 delay4)
[[ -n ${CONDS:-} ]] && read -r -a ORDER <<< "$CONDS"

LOG=$RUN/ood_logs_${TAG}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$LOG"
echo "logs: $LOG"
run_one() {
  local name=$1; shift
  # shellcheck disable=SC2086
  "${EVAL[@]}" --tag "${TAG}_${name}" ${COND[$name]} "$@" > "$LOG/$name.log" 2>&1 \
    && echo "done  $name" || echo "FAIL  $name (see $LOG/$name.log)"
}
n=0
for name in "${ORDER[@]}"; do
  run_one "$name" "$@" &
  n=$((n + 1))
  if (( n % JOBS == 0 )); then wait; fi
done
wait

# collect the eval dirs written by this batch (path is printed last by the evaluator)
ARGS=()
for name in "${ORDER[@]}"; do
  d=$(grep -h "^saved: " "$LOG/$name.log" | tail -1 | sed 's/^saved: //')
  [[ -n $d ]] && ARGS+=(--eval-dir "$d" --label "$name")
done
python "$REPO/fm_pc/ood_metrics.py" "${ARGS[@]}" --out "$LOG/summary"
