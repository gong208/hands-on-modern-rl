"""Final held-out evaluation of a trained checkpoint on the GSM8k test set.

Steps:
  1. Locate the latest ``global_step_*/actor`` checkpoint under a run directory.
  2. Merge the sharded FSDP checkpoint into a standalone HuggingFace model via
     ``verl.model_merger`` (the same tool the chapter 8 project uses to produce
     its ``merged_model/`` artifact).
  3. Run greedy vLLM inference over the GSM8k test split and score answers with
     the *same* ``extract_answer`` / ``check_answer`` used by the training
     reward, so the eval metric is exactly consistent with what was optimized.

Why not evalscope? evalscope is installed in the image and is a fine
standardized harness, but it adds dataset-download/config surface area and uses
its own answer matcher. Reusing the training reward's matcher keeps the PPO-vs-
GRPO comparison apples-to-apples. To use evalscope instead, see the README.

Usage (inside the container):
  python3 eval_final.py --run-dir results/ppo --out results/ppo/eval.json
  python3 eval_final.py --run-dir results/grpo --out results/grpo/eval.json --limit 64
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from reward import check_answer, extract_answer


def find_latest_actor_checkpoint(run_dir: Path) -> Path:
    """Return the actor dir of the highest global_step_* under run_dir/checkpoints."""
    ckpt_root = run_dir / "checkpoints"
    steps = sorted(
        ckpt_root.glob("global_step_*"),
        key=lambda p: int(re.search(r"global_step_(\d+)", p.name).group(1)),
    )
    if not steps:
        raise FileNotFoundError(f"No global_step_* checkpoints under {ckpt_root}")
    actor = steps[-1] / "actor"
    if not actor.exists():
        raise FileNotFoundError(f"No actor/ dir in latest checkpoint {steps[-1]}")
    return actor


def merge_to_hf(actor_dir: Path, target_dir: Path) -> Path:
    """Merge FSDP shards into a standalone HF model (idempotent)."""
    import subprocess

    if (target_dir / "config.json").exists():
        print(f"[eval] merged model already present at {target_dir}, skipping merge")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(actor_dir),
        "--target_dir", str(target_dir),
    ]
    print("[eval] merging:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return target_dir


def evaluate(model_dir: Path, test_file: Path, limit: int | None, max_new_tokens: int):
    """Greedy vLLM generation over the test split; score with the training matcher."""
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(test_file)
    if limit is not None:
        df = df.iloc[:limit]

    # prompt column is a list of chat messages; ground truth lives in reward_model.
    messages = [list(p) for p in df["prompt"].tolist()]
    ground_truths = [r["ground_truth"] for r in df["reward_model"].tolist()]

    llm = LLM(model=str(model_dir), gpu_memory_utilization=0.6, dtype="bfloat16")
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    t0 = time.time()
    outputs = llm.chat(messages, sampling)
    elapsed = time.time() - t0

    n_correct = 0
    total_out_tokens = 0
    total_in_tokens = 0
    records = []
    for out, gt in zip(outputs, ground_truths):
        text = out.outputs[0].text
        total_out_tokens += len(out.outputs[0].token_ids)
        total_in_tokens += len(out.prompt_token_ids)
        predicted = extract_answer(text)
        correct = check_answer(predicted, gt) >= 1.0
        n_correct += int(correct)
        records.append({"predicted": predicted, "ground_truth": gt, "correct": correct})

    n = len(records)
    return {
        "n_samples": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        # Inference profile (batched vLLM generation over the test set).
        "eval_seconds": round(elapsed, 2),
        "avg_output_tokens": round(total_out_tokens / n, 1) if n else None,
        "avg_input_tokens": round(total_in_tokens / n, 1) if n else None,
        "total_output_tokens": total_out_tokens,
        "decode_throughput_tok_s": round(total_out_tokens / elapsed, 1) if elapsed else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Exactly one of --run-dir (a trained run, merged from its latest checkpoint)
    # or --model-dir (a ready HF model, e.g. the untrained base) must be given.
    ap.add_argument("--run-dir", type=Path,
                    help="Run directory containing checkpoints/ (e.g. results/ppo)")
    ap.add_argument("--model-dir", type=Path,
                    help="A ready HF model dir to evaluate directly, skipping "
                         "checkpoint-find/merge (e.g. the base model). Use this to "
                         "evaluate the base model under the SAME 0-shot harness as "
                         "PPO/GRPO for an apples-to-apples Base-vs-PPO-vs-GRPO table.")
    ap.add_argument("--test-file", type=Path, default=Path("/root/data/gsm8k/test.parquet"))
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    ap.add_argument("--limit", type=int, default=None, help="Eval only first N (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    if bool(args.run_dir) == bool(args.model_dir):
        ap.error("provide exactly one of --run-dir or --model-dir")

    if args.model_dir:
        model_dir = args.model_dir
        result = evaluate(model_dir, args.test_file, args.limit, args.max_new_tokens)
        result["model_dir"] = str(model_dir)
        label = model_dir.name
    else:
        actor_dir = find_latest_actor_checkpoint(args.run_dir)
        model_dir = merge_to_hf(actor_dir, args.run_dir / "merged_model")
        result = evaluate(model_dir, args.test_file, args.limit, args.max_new_tokens)
        result["run_dir"] = str(args.run_dir)
        result["checkpoint"] = str(actor_dir)
        label = args.run_dir.name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[eval] {label}: "
          f"accuracy={result['accuracy']:.4f} "
          f"({result['n_correct']}/{result['n_samples']}) -> {args.out}")


if __name__ == "__main__":
    main()
