#!/usr/bin/env bash
# OOD benchmark: ID + OOD-1..4 for one task and one method, then ood_metrics.py.
#
#   bash fm_pc/run_ood_benchmark.sh pushing                       # baseline policy
#   bash fm_pc/run_ood_benchmark.sh pushing --sampler pc --constraints action_box,tracking
#   TAG=pc QUICK=1 bash fm_pc/run_ood_benchmark.sh inserting --sampler pc ...
#
# Extra arguments are passed to the evaluator (method flags).  Env vars:
#   TAG       prefix of every run tag (default: base)
#   QUICK=1   small smoke run (Pushing 6 contexts x 1, Inserting 4 contexts x 1)
#   JOBS      evaluations in parallel (default 1)
#   CONDS     space-separated subset of condition names below (default: all)
#   SEED      policy seed (default 0); the OOD draws use --ood-seed 0
set -euo pipefail
TASK=${1:?usage: run_ood_benchmark.sh pushing|inserting [method flags...]}
shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG=${TAG:-base}
JOBS=${JOBS:-1}
SEED=${SEED:-0}

if [[ $TASK == pushing ]]; then
  RUN=$REPO/fm_baseline/results/pushing_baseline
  EVAL=(python "$REPO/fm_pc/eval_pushing.py" --run-dir "$RUN" --ckpt baseline_ema.pt --execute-steps 8 --seed "$SEED")
  if [[ ${QUICK:-0} == 1 ]]; then EVAL+=(--n-contexts 6 --n-traj 1); else EVAL+=(--n-contexts 30 --n-traj 4); fi
elif [[ $TASK == inserting ]]; then
  RUN=$REPO/fm_baseline/results/inserting_baseline
  EVAL=(python "$REPO/fm_pc/eval_inserting.py" --run-dir "$RUN" --ckpt baseline_ema.pt --execute-steps 8 --seed "$SEED")
  if [[ ${QUICK:-0} == 1 ]]; then EVAL+=(--n-contexts 4 --n-traj 1); else EVAL+=(--n-traj 4); fi
else
  echo "task must be pushing or inserting" >&2; exit 1
fi

# name | OOD flags   (magnitudes follow the 2026-11-05 slides; edit here, not in the code)
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
