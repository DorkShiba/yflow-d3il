#!/usr/bin/env bash
# Run from any directory, after activating the environment in docs/stacking_team_run.md.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python -m fm_constraint.check_stacking_bundle
export MUJOCO_GL="${MUJOCO_GL:-disable}"
python -m fm_constraint.eval_stacking --settings configs/stacking_combo31.json "$@"
