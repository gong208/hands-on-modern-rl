# Shared configuration for the PPO-vs-GRPO controlled comparison.
#
# Sourced by run_ppo.sh and run_grpo.sh. Every value here is held IDENTICAL
# across both runs so that the only difference between the two trainings is the
# advantage estimator (PPO = gae + critic, GRPO = grpo + group baseline).
#
# All values use ${VAR:-default} so they can be overridden from the environment
# (e.g. for the smoke test: TOTAL_EPOCHS=1 TRAIN_BATCH_SIZE=16 bash run_ppo.sh).

# ── Paths (verified present inside verl-gsm8k:latest) ─────────────────────────
export MODEL_PATH=${MODEL_PATH:-/opt/data/model_scope/Qwen/Qwen2.5-0.5B-Instruct}
export GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-/root/data/gsm8k/train.parquet}
export GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-/root/data/gsm8k/test.parquet}

# ── Hardware: single GPU, sequential runs ─────────────────────────────────────
export NNODES=${NNODES:-1}
export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}

# ── Identical optimization / data budget for both methods ─────────────────────
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-5}

# ── Equal sampling: same group size for BOTH runs ─────────────────────────────
# This is the fairness control. PPO normally uses n=1; here both draw n samples
# per prompt so the only remaining difference is how the advantage is computed.
export ROLLOUT_N=${ROLLOUT_N:-4}

# ── KL regularization, held equal across both runs ────────────────────────────
# Applied as an actor-side KL loss against the reference model for BOTH methods,
# so KL handling is not a confound. use_kl_in_reward stays off for both.
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
export KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}

# ── Shared reward (single source of truth) ────────────────────────────────────
SCRIPT_DIR_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-$SCRIPT_DIR_COMMON/reward.py}
export REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:-compute_score}

# ── Bookkeeping ───────────────────────────────────────────────────────────────
# Log to wandb locally (offline) by default: keeps metric artifacts without
# requiring an API key/network, and avoids noisy connection errors at teardown.
# Set WANDB_MODE=online and WANDB_API_KEY to sync to the wandb cloud.
export WANDB_MODE=${WANDB_MODE:-offline}
export PROJECT_NAME=${PROJECT_NAME:-ppo_vs_grpo_gsm8k}
export RESULTS_DIR=${RESULTS_DIR:-$SCRIPT_DIR_COMMON/results}
# Used by collect_metrics.py to turn GPU-hours into an estimated dollar cost.
export L40S_HOURLY_RATE=${L40S_HOURLY_RATE:-1.50}
