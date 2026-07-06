#!/bin/bash
# GRPO arm of the comparison (advantage estimator = grpo, NO critic network).
#
# Mirrors the chapter 8 PPO launcher block-for-block, with the algorithmic
# differences that define GRPO:
#   - algorithm.adv_estimator=grpo   (group baseline instead of GAE)
#   - the entire CRITIC block is removed (no value network to train)
#   - actor-side KL loss against the reference model
# Everything else (model, data, batch sizes, lengths, LR, rollout.n, epochs,
# reward) is identical to the PPO arm via common_env.sh, so the only difference
# is how the advantage is estimated.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo}
# Per-run output dir; checkpoints land in a deterministic place for eval/merge.
RUN_DIR=${RUN_DIR:-$RESULTS_DIR/grpo}
mkdir -p "$RUN_DIR/checkpoints"

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files="['$GSM8K_TRAIN_FILE']"
    data.val_files="['$GSM8K_TEST_FILE']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# GRPO-specific actor settings: group baseline needs no critic, KL is applied
# as an actor-side loss against the reference policy.
ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# NOTE: no CRITIC block — GRPO has no value network.

REWARD=(
    custom_reward_function.path="$REWARD_FUNCTION_PATH"
    custom_reward_function.name="$REWARD_FUNCTION_NAME"
)

ALGORITHM=(
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="$RUN_DIR/checkpoints"
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${ALGORITHM[@]}" \
    "${TRAINER[@]}" \
    "$@"
