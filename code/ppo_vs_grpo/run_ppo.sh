#!/bin/bash
# PPO arm of the comparison (advantage estimator = GAE, with a critic network).
#
# This is a thin wrapper: it loads the shared config and then delegates to the
# existing, battle-tested chapter 8 PPO launcher, which is already fully
# env-parametrized and defaults to algorithm.adv_estimator=gae with the full
# critic block. We only override the experiment name and add the KL-loss term so
# that KL handling matches the GRPO arm (the only intended difference is the
# advantage estimator). Any extra args are forwarded to the launcher.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo_gae}
# Per-run output dir; checkpoints land in a deterministic place for eval/merge.
RUN_DIR=${RUN_DIR:-$RESULTS_DIR/ppo}
mkdir -p "$RUN_DIR/checkpoints"

PPO_LAUNCHER="$SCRIPT_DIR/../chapter08_rlhf/verl_gsm8k/run_qwen2_5_0_5b_ppo_single_gpu.sh"

# Match the GRPO arm's KL treatment: actor-side KL loss against the ref model.
# (project_name is intentionally not overridden here — the chapter 8 launcher
# already sets it, and Hydra rejects a duplicate override.)
OVERRIDES=(
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}
    algorithm.use_kl_in_reward=False
    trainer.default_local_dir="$RUN_DIR/checkpoints"
)

bash "$PPO_LAUNCHER" "${OVERRIDES[@]}" "$@"
