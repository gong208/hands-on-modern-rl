# PPO vs GRPO: GSM8K Controlled Experiment

This directory contains a controlled PPO-vs-GRPO comparison on GSM8K using
veRL. Both runs train the same base model on the same data with the same reward,
batching, rollout count, KL setup, and single-GPU environment. The intended
algorithmic difference is the advantage estimator:

- PPO: `algorithm.adv_estimator=gae` with a critic.
- GRPO: `algorithm.adv_estimator=grpo` without a critic.

The experiment answers a narrow question: under matched training conditions, how
much accuracy does GRPO retain, and how much training cost does it save, compared
with PPO?

## Result Summary

The completed run used Qwen2.5-0.5B-Instruct on GSM8K, 20 epochs per method,
`rollout.n=4`, one L40S GPU, and the shared reward in `reward.py`. Final test
evaluation uses greedy vLLM decoding plus the same answer matcher used by the
training reward.

| Model | GSM8K test accuracy | vs base |
| --- | ---: | ---: |
| Base Qwen2.5-0.5B-Instruct | 45.19% | - |
| PPO, GAE + critic | 58.83% | +13.6 pts |
| GRPO, no critic | 57.47% | +12.3 pts |

| Metric | PPO | GRPO | GRPO vs PPO |
| --- | ---: | ---: | ---: |
| Final test accuracy | 58.83% | 57.47% | -2.3% |
| Best in-training validation score | 0.5914 | 0.5876 | -0.6% |
| Wall-clock time | 8.47 h | 6.49 h | -23.4% |
| GPU-hours | 8.47 | 6.49 | -23.4% |
| Estimated cost at `$1.50/GPU-hour` | `$12.71` | `$9.74` | -23.4% |
| Peak GPU memory | 47.36 GB | 47.31 GB | -0.1% |
| Mean GPU utilization | 72.1% | 67.1% | -6.9% |
| Median training throughput | 6138 tok/s | 8192 tok/s | +33.5% |

Takeaway: in this run, GRPO reached nearly the same GSM8K accuracy as PPO while
using about 23% fewer GPU-hours and achieving higher median training throughput.
Peak device memory was almost unchanged in the external `nvidia-smi` trace,
although veRL's internal allocated-memory metric was lower for GRPO.

The full generated report is written to
`results/comparison_report.{json,md}` when `run_comparison.sh` finishes. The
`results/` and `wandb/` directories are intentionally ignored by git because
they contain logs, checkpoints, merged models, and binary wandb artifacts.

## What Is Controlled

| Held identical through `common_env.sh` | PPO arm | GRPO arm |
| --- | --- | --- |
| Base model: Qwen2.5-0.5B-Instruct | GAE advantage estimator | GRPO group baseline |
| Dataset: GSM8K train/test parquet files | Critic enabled | No critic block |
| Reward: shared `reward.py` | Actor-side KL loss | Actor-side KL loss |
| Batch sizes, learning rate, epochs, sequence lengths | `rollout.n=4` | `rollout.n=4` |
| Single-GPU sequential execution | veRL PPO launcher | veRL GRPO launcher |

Using the same rollout sample count for both methods is deliberate. PPO often
runs with `rollout.n=1`, but this experiment sets both arms to `rollout.n=4` so
sampling budget is not a confound.

## Files

| File | Purpose |
| --- | --- |
| `common_env.sh` | Shared hyperparameters and paths; every value is env-overridable. |
| `reward.py` | Re-exports the chapter 8 GSM8K reward and answer matcher. |
| `run_ppo.sh` | PPO arm: delegates to the chapter 8 veRL PPO launcher with matched KL settings. |
| `run_grpo.sh` | GRPO arm: mirrors the PPO setup with `adv_estimator=grpo` and no critic. |
| `gpu_monitor.sh` | Background `nvidia-smi` sampler that writes `gpu_mem.csv`. |
| `eval_final.py` | Merges the latest FSDP actor checkpoint and evaluates with vLLM. |
| `collect_metrics.py` | Aggregates logs, GPU samples, timing metadata, and eval results into a report. |
| `run_comparison.sh` | End-to-end orchestrator: PPO, GRPO, final eval, then report generation. |

## Environment

The scripts are intended to run inside the prebuilt `verl-gsm8k:latest` image.
That image is expected to contain veRL, vLLM, the base model, and GSM8K parquet
files at the default paths configured in `common_env.sh`:

- `MODEL_PATH=/opt/data/model_scope/Qwen/Qwen2.5-0.5B-Instruct`
- `GSM8K_TRAIN_FILE=/root/data/gsm8k/train.parquet`
- `GSM8K_TEST_FILE=/root/data/gsm8k/test.parquet`

Important defaults:

| Variable | Default |
| --- | --- |
| `TOTAL_EPOCHS` | `20` |
| `TRAIN_BATCH_SIZE` | `128` |
| `PPO_MINI_BATCH_SIZE` | `64` |
| `ROLLOUT_N` | `4` |
| `KL_LOSS_COEF` | `0.001` |
| `SAVE_FREQ` | `20` |
| `TEST_FREQ` | `5` |
| `L40S_HOURLY_RATE` | `1.50` |

Override any of these from the environment before running a script.

## Run The Full Comparison

From the repository root:

```bash
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v /root/hands-on-modern-rl:/workspace \
  -v /opt/data:/opt/data \
  -w /workspace/code/ppo_vs_grpo \
  verl-gsm8k:latest \
  bash run_comparison.sh
```

This runs PPO first, then GRPO, on the same visible GPU. Sequential execution is
intentional: it avoids cross-run contention in CPU, memory, disk, and GPU
metrics.

## Smoke Test

Use this to verify the pipeline quickly. It is not a meaningful comparison:

```bash
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v /root/hands-on-modern-rl:/workspace \
  -v /opt/data:/opt/data \
  -w /workspace/code/ppo_vs_grpo \
  verl-gsm8k:latest \
  bash -c 'TOTAL_EPOCHS=1 TRAIN_BATCH_SIZE=16 SAVE_FREQ=1 TEST_FREQ=1 EVAL_LIMIT=32 bash run_comparison.sh'
```

## Run One Arm

Inside the container, from this directory:

```bash
RUN_DIR=results/ppo bash run_ppo.sh
RUN_DIR=results/grpo bash run_grpo.sh
```

After a single-arm run, evaluate it manually:

```bash
python3 eval_final.py \
  --run-dir results/ppo \
  --test-file "$GSM8K_TEST_FILE" \
  --out results/ppo/eval.json
```

Evaluate the untrained base model with the same harness:

```bash
python3 eval_final.py \
  --model-dir "$MODEL_PATH" \
  --test-file "$GSM8K_TEST_FILE" \
  --out results/base/eval.json
```

Regenerate the comparison report:

```bash
python3 collect_metrics.py \
  --results-dir results \
  --hourly-rate "$L40S_HOURLY_RATE"
```

## Outputs

Expected output layout:

```text
results/
├── base/
│   └── eval.json
├── ppo/
│   ├── checkpoints/
│   ├── merged_model/
│   ├── train.log
│   ├── gpu_mem.csv
│   ├── run_meta.json
│   └── eval.json
├── grpo/
│   └── ...
├── comparison_report.json
└── comparison_report.md
```

`comparison_report.md` is the human-readable artifact to inspect first. The JSON
file preserves the same metrics for downstream analysis.

## Notes

- Accuracy is scored with the same `extract_answer` and `check_answer` functions
  used by the training reward.
- The dollar estimate is only `GPU-hours * L40S_HOURLY_RATE`; set that variable
  to match the actual machine cost.
- `collect_metrics.py` is defensive: missing logs or eval files become `null`
  values in the report instead of crashing.
- The committed source intentionally excludes generated checkpoints, logs,
  merged models, wandb artifacts, and Python bytecode.
