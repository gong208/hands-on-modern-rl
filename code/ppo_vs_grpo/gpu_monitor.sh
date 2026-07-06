#!/bin/bash
# Background GPU sampler. Polls nvidia-smi at a fixed interval and appends one
# CSV row per sample for the pinned GPU. collect_metrics.py reads this to report
# peak GPU memory (and a utilization trace) for each run.
#
# Usage:
#   bash gpu_monitor.sh <out_csv> [gpu_index] [interval_seconds]
#
# Run it in the background and kill it when the training finishes, e.g.:
#   bash gpu_monitor.sh results/ppo/gpu_mem.csv 0 2 &
#   MON_PID=$!
#   ... training ...
#   kill "$MON_PID"
set -euo pipefail

OUT_CSV=${1:?"usage: gpu_monitor.sh <out_csv> [gpu_index] [interval_seconds]"}
GPU_INDEX=${2:-0}
INTERVAL=${3:-2}

mkdir -p "$(dirname "$OUT_CSV")"
echo "timestamp,gpu_index,mem_used_mib,mem_total_mib,util_gpu_pct" > "$OUT_CSV"

# --loop runs nvidia-smi natively at INTERVAL; --format=csv,noheader keeps rows
# clean. We prepend an ISO timestamp ourselves so samples are wall-clock anchored.
while true; do
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    line=$(nvidia-smi -i "$GPU_INDEX" \
        --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
    if [ -n "$line" ]; then
        echo "${ts},${GPU_INDEX},${line}" >> "$OUT_CSV"
    fi
    sleep "$INTERVAL"
done
