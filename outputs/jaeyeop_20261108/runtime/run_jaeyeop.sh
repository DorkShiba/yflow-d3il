#!/usr/bin/env bash
set -uo pipefail
cd /root/d3il-work/yflow-d3il
export PYTHON=/root/d3il-work/venv/bin/python
export PATH=/root/d3il-work/venv/bin:$PATH
export MUJOCO_GL=disable OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_root=/root/d3il-work/jaeyeop_20261108
mkdir -p "$run_root"
date -u +%FT%TZ > "$run_root/started_utc.txt"
git rev-parse HEAD > "$run_root/commit.txt"
pids=()
for ctx in 17 18 19; do
  (
    rc=0
    # Measure a complete Inserting baseline unit early; the normal run later skips it.
    if [[ "$ctx" == 17 ]]; then
      bash scripts/run_experiment.sh --name jaeyeop --contexts "$ctx" --task inserting --combos 0 --device cuda:0 > "$run_root/ctx${ctx}_inserting_baseline.log" 2>&1 || rc=$?
    fi
    if [[ "$rc" == 0 ]]; then
      bash scripts/run_experiment.sh --name jaeyeop --contexts "$ctx" --device cuda:0 > "$run_root/ctx${ctx}.log" 2>&1 || rc=$?
    fi
    printf '%s\n' "$rc" > "$run_root/ctx${ctx}.exitcode"
    date -u +%FT%TZ > "$run_root/ctx${ctx}.finished_utc"
    exit "$rc"
  ) &
  pids+=("$!")
  printf '%s %s\n' "$ctx" "$!" >> "$run_root/worker_pids.txt"
  sleep 3
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" == 0 ]]; then
  "$PYTHON" -m fm_constraint.experiment report --name jaeyeop > "$run_root/report.log" 2>&1 || failed=1
fi
printf '%s\n' "$failed" > "$run_root/exitcode"
date -u +%FT%TZ > "$run_root/finished_utc.txt"
exit "$failed"
