#!/bin/bash
# End-to-end PPO-vs-GRPO comparison orchestrator. Run INSIDE the container.
#
#   docker run --rm --gpus '"device=0"' --shm-size=16g \
#     -v /root/hands-on-modern-rl:/workspace \
#     -v /opt/data:/opt/data \
#     -w /workspace/code/ppo_vs_grpo verl-gsm8k:latest \
#     bash run_comparison.sh
#
# Runs PPO then GRPO sequentially on a single pinned GPU (clean, uncontended
# cost numbers), evaluates each final checkpoint on the GSM8k test set, and
# writes results/comparison_report.{json,md}.
#
# Smoke test (fast sanity check, not a real comparison):
#   TOTAL_EPOCHS=1 TRAIN_BATCH_SIZE=16 SAVE_FREQ=1 TEST_FREQ=1 EVAL_LIMIT=32 \
#     bash run_comparison.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

# Pin to a single GPU for the whole comparison.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
GPU_INDEX=0   # always 0 within the masked view
EVAL_LIMIT=${EVAL_LIMIT:-}

mkdir -p "$RESULTS_DIR"

run_arm() {
    local name="$1"          # ppo | grpo
    local launcher="$2"      # path to run_ppo.sh / run_grpo.sh
    local run_dir="$RESULTS_DIR/$name"
    mkdir -p "$run_dir"

    echo "=================================================================="
    echo "  [$name] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "=================================================================="

    # Background GPU sampler.
    bash "$SCRIPT_DIR/gpu_monitor.sh" "$run_dir/gpu_mem.csv" "$GPU_INDEX" 2 &
    local mon_pid=$!

    local start_s end_s rc
    start_s=$(date +%s)
    # Don't let a training failure abort the whole comparison.
    RUN_DIR="$run_dir" EXPERIMENT_NAME="$name" \
        bash "$launcher" 2>&1 | tee "$run_dir/train.log"
    rc=${PIPESTATUS[0]}
    end_s=$(date +%s)

    kill "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true

    # Persist timing for collect_metrics.py.
    python3 - "$run_dir/run_meta.json" "$name" "$start_s" "$end_s" "$rc" <<'PY'
import json, sys
out, name, start_s, end_s, rc = sys.argv[1:6]
wall = int(end_s) - int(start_s)
json.dump({
    "run": name,
    "start_ts": int(start_s),
    "end_ts": int(end_s),
    "wall_clock_seconds": wall,
    "gpus": 1,
    "exit_code": int(rc),
}, open(out, "w"), indent=2)
print(f"[{name}] wall-clock {wall}s, exit code {rc}")
PY

    if [ "$rc" -ne 0 ]; then
        echo "[$name] WARNING: training exited with code $rc — continuing."
    fi
}

eval_arm() {
    local name="$1"
    local run_dir="$RESULTS_DIR/$name"
    echo "------------------------------------------------------------------"
    echo "  [$name] final GSM8k test eval"
    echo "------------------------------------------------------------------"
    local limit_args=()
    [ -n "$EVAL_LIMIT" ] && limit_args=(--limit "$EVAL_LIMIT")
    python3 "$SCRIPT_DIR/eval_final.py" \
        --run-dir "$run_dir" \
        --test-file "$GSM8K_TEST_FILE" \
        --out "$run_dir/eval.json" \
        "${limit_args[@]}" \
        || echo "[$name] eval failed — report will show n/a for accuracy."
}

# ── Sequential training: PPO then GRPO on the same GPU ────────────────────────
run_arm  ppo  "$SCRIPT_DIR/run_ppo.sh"
run_arm  grpo "$SCRIPT_DIR/run_grpo.sh"

# ── Final held-out evaluation ─────────────────────────────────────────────────
eval_arm ppo
eval_arm grpo

# ── Aggregate into the comparison report ──────────────────────────────────────
python3 "$SCRIPT_DIR/collect_metrics.py" \
    --results-dir "$RESULTS_DIR" \
    --hourly-rate "$L40S_HOURLY_RATE"

echo "Done. See $RESULTS_DIR/comparison_report.md"
