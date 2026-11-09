#!/usr/bin/env bash
# 준혁 담당 ID 본실험(README 11절: context 7-16, 32조합, Pushing→Inserting)을 GPU별 프로세스로 띄운다.
# run_experiment.sh는 단위마다 잠금을 잡으므로 같은 범위를 여러 프로세스가 나눠 처리한다.
# 컨테이너 CPU 6개 → 프로세스 6개. GPU 4장이라 0·1번에 2개씩.
#
#   bash /workspace/launch_id_v1.sh            # 시작
#   bash /workspace/launch_id_v1.sh status     # 진행 상황
#   중단 후 같은 명령을 다시 실행하면 끝난 단위는 건너뛴다.
set -euo pipefail
REPO=/workspace/yflow-d3il-main
export PYTHON=/workspace/py311/bin/python
DEVICES=(0 1 2 3 0 1)
cd "$REPO"

if [[ "${1:-}" == status ]]; then
    set +e +o pipefail
    echo "running: $(pgrep -fc 'fm_constraint.experiment run' || true) processes"
    for t in pushing inserting; do
        n=$(find experiments/id_v1/$t -name result.json 2>/dev/null | xargs -r grep -l '"member": "junhyuk"' | wc -l)
        echo "$t: $n / 320 units done"
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    tail -n 2 fm_constraint/runs/junhyuk_cuda*.out 2>/dev/null | grep -E "DONE|RUN|BUSY|Error|Traceback" | tail -6 || true
    exit 0
fi

if pgrep -f 'fm_constraint.experiment run' >/dev/null; then
    echo "이미 실행 중입니다. 진행 상황: bash $0 status" >&2
    exit 1
fi
mkdir -p fm_constraint/runs
for d in "${DEVICES[@]}"; do
    out="fm_constraint/runs/junhyuk_cuda${d}_$(date +%Y%m%dT%H%M%S%N).out"
    nohup bash scripts/run_experiment.sh --name junhyuk --contexts 7-16 --device "cuda:$d" > "$out" 2>&1 &
    echo "started cuda:$d -> $out"
    sleep 10
done
echo "진행 상황: bash $0 status"
