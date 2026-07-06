"""Aggregate per-run artifacts into a single PPO-vs-GRPO comparison report.

For each run directory (results/ppo, results/grpo) this reads, defensively:
  - run_meta.json   : wall-clock timing + GPU count (written by run_comparison.sh)
  - gpu_mem.csv     : nvidia-smi samples (written by gpu_monitor.sh) -> peak memory
  - train.log       : veRL console log -> throughput (tokens/s) + best val score
  - eval.json       : final held-out GSM8k test accuracy (written by eval_final.py)

Emits:
  - results/comparison_report.json  (machine-readable)
  - results/comparison_report.md    (side-by-side table with deltas)

Any missing artifact degrades to null rather than failing, so partial runs still
produce a readable report.
"""

import argparse
import csv
import json
import re
from pathlib import Path

# Patterns are intentionally loose: veRL's console logger formats metrics as a
# dict, so keys appear as either key:value or 'key': value. Numeric values may be
# bare (e.g. perf/throughput:2525.3) or wrapped in a numpy repr
# (e.g. val-core/.../reward/mean@1:np.float64(0.582)), so the value capture
# tolerates an optional np.float64( prefix.
_NUM = r"(?:np\.float64\()?\s*([0-9.]+)"
_THROUGHPUT_RE = re.compile(r"perf/throughput['\"]?\s*[:=]\s*" + _NUM)
_VAL_RE = re.compile(
    r"(val[\w\-/@.]*?(?:score|acc|reward)[\w\-/@.]*?)['\"]?\s*[:=]\s*" + _NUM
)
_MAXMEM_RE = re.compile(r"perf/max_memory_allocated_gb['\"]?\s*[:=]\s*" + _NUM)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def peak_mem_mib(gpu_csv: Path):
    """Max mem_used_mib across all samples, plus mean GPU utilization."""
    try:
        rows = list(csv.DictReader(gpu_csv.open()))
    except FileNotFoundError:
        return None, None
    if not rows:
        return None, None
    mems = [float(r["mem_used_mib"]) for r in rows if r.get("mem_used_mib")]
    utils = [float(r["util_gpu_pct"]) for r in rows if r.get("util_gpu_pct")]
    peak = max(mems) if mems else None
    mean_util = round(sum(utils) / len(utils), 1) if utils else None
    return peak, mean_util


def parse_train_log(log_path: Path):
    """Best-effort throughput (median + last) and best val score from the log."""
    try:
        text = log_path.read_text(errors="ignore")
    except FileNotFoundError:
        return {"throughput_tokens_s_median": None,
                "throughput_tokens_s_last": None,
                "best_val_score": None,
                "max_memory_allocated_gb": None}

    tputs = [float(x) for x in _THROUGHPUT_RE.findall(text)]
    tputs_sorted = sorted(tputs)
    median_tput = tputs_sorted[len(tputs_sorted) // 2] if tputs_sorted else None

    val_scores = [float(v) for _, v in _VAL_RE.findall(text)]
    best_val = max(val_scores) if val_scores else None

    maxmems = [float(x) for x in _MAXMEM_RE.findall(text)]
    max_mem_gb = max(maxmems) if maxmems else None

    return {
        "throughput_tokens_s_median": median_tput,
        "throughput_tokens_s_last": tputs[-1] if tputs else None,
        "best_val_score": best_val,
        "max_memory_allocated_gb": max_mem_gb,
    }


def collect_run(run_dir: Path, hourly_rate: float) -> dict:
    meta = _load_json(run_dir / "run_meta.json") or {}
    evalj = _load_json(run_dir / "eval.json") or {}
    peak_mib, mean_util = peak_mem_mib(run_dir / "gpu_mem.csv")
    logm = parse_train_log(run_dir / "train.log")

    wall_s = meta.get("wall_clock_seconds")
    gpus = meta.get("gpus", 1)
    gpu_hours = round((wall_s / 3600.0) * gpus, 4) if wall_s else None
    est_cost = round(gpu_hours * hourly_rate, 2) if gpu_hours is not None else None

    return {
        "run": run_dir.name,
        "exit_code": meta.get("exit_code"),
        # performance
        "final_test_accuracy": evalj.get("accuracy"),
        "test_correct": evalj.get("n_correct"),
        "test_samples": evalj.get("n_samples"),
        "best_val_score": logm["best_val_score"],
        # inference profile (from eval_final.py's batched vLLM generation)
        "eval_seconds": evalj.get("eval_seconds"),
        "avg_output_tokens": evalj.get("avg_output_tokens"),
        "avg_input_tokens": evalj.get("avg_input_tokens"),
        "decode_throughput_tok_s": evalj.get("decode_throughput_tok_s"),
        # cost
        "wall_clock_seconds": wall_s,
        "wall_clock_hours": round(wall_s / 3600.0, 4) if wall_s else None,
        "gpus": gpus,
        "gpu_hours": gpu_hours,
        "estimated_usd": est_cost,
        # efficiency
        "peak_gpu_mem_mib": peak_mib,
        "peak_gpu_mem_gb": round(peak_mib / 1024.0, 2) if peak_mib else None,
        "max_memory_allocated_gb_log": logm["max_memory_allocated_gb"],
        "mean_gpu_util_pct": mean_util,
        "throughput_tokens_s_median": logm["throughput_tokens_s_median"],
        "throughput_tokens_s_last": logm["throughput_tokens_s_last"],
    }


def _fmt(v, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:g}{suffix}"
    return f"{v}{suffix}"


def _delta(grpo, ppo):
    """GRPO relative to PPO, as a signed percentage where both are present."""
    if grpo is None or ppo is None or ppo == 0:
        return "n/a"
    return f"{(grpo - ppo) / abs(ppo) * 100:+.1f}%"


def _acc_pct(run: dict) -> str:
    """Final test accuracy as a percentage string, or n/a."""
    a = run.get("final_test_accuracy")
    return f"{a * 100:.2f}%" if a is not None else "n/a"


def _pts_vs_base(run: dict, base: dict | None) -> str:
    """Accuracy gain over the base model, in percentage points."""
    a, b = run.get("final_test_accuracy"), (base or {}).get("final_test_accuracy")
    if a is None or b is None:
        return "n/a"
    return f"{(a - b) * 100:+.1f} pts"


def render_md(ppo: dict, grpo: dict, hourly_rate: float, base: dict | None = None) -> str:
    rows = [
        ("Final GSM8k test accuracy", "final_test_accuracy", ""),
        ("Best in-training val score", "best_val_score", ""),
        ("Wall-clock (hours)", "wall_clock_hours", " h"),
        ("GPU-hours", "gpu_hours", ""),
        (f"Estimated cost (@${hourly_rate}/GPU-h)", "estimated_usd", " $"),
        ("Peak GPU memory (GB)", "peak_gpu_mem_gb", " GB"),
        ("Mean GPU utilization", "mean_gpu_util_pct", " %"),
        ("Throughput (tok/s, median)", "throughput_tokens_s_median", ""),
    ]
    lines = [
        "# PPO vs GRPO — GSM8k Comparison Report",
        "",
        "Same model (Qwen2.5-0.5B-Instruct), same dataset (GSM8k), same reward, "
        "same batch/epochs/LR/rollout.n, same KL. The only difference is the "
        "advantage estimator: **PPO** = GAE + critic, **GRPO** = group baseline "
        "(no critic).",
        "",
    ]

    # Three-way accuracy table (Base vs PPO vs GRPO), all scored through the same
    # 0-shot eval harness. Only shown when a base eval (results/base/eval.json) is
    # present; otherwise we skip it rather than print an empty table.
    if base and base.get("final_test_accuracy") is not None:
        lines += [
            "## Accuracy — Base vs PPO vs GRPO",
            "",
            "All three scored through the identical 0-shot harness "
            "(`eval_final.py`: greedy, same answer matcher as the training reward).",
            "",
            "| Model | GSM8k test accuracy | vs base |",
            "| --- | --- | --- |",
            f"| Qwen2.5-0.5B-Instruct (base) | {_acc_pct(base)} | — |",
            f"| PPO (GAE + critic) | {_acc_pct(ppo)} | {_pts_vs_base(ppo, base)} |",
            f"| GRPO (no critic) | {_acc_pct(grpo)} | {_pts_vs_base(grpo, base)} |",
            "",
            "## PPO vs GRPO — full metrics",
            "",
        ]

    lines += [
        "| Metric | PPO (gae) | GRPO | GRPO vs PPO |",
        "| --- | --- | --- | --- |",
    ]
    for label, key, suffix in rows:
        p, g = ppo.get(key), grpo.get(key)
        lines.append(f"| {label} | {_fmt(p, suffix)} | {_fmt(g, suffix)} | {_delta(g, p)} |")

    # Inference profile of the FINAL model on the test set (batched vLLM gen).
    # One row per model; base included when present.
    inf_models = []
    if base and base.get("final_test_accuracy") is not None:
        inf_models.append(("Qwen2.5-0.5B-Instruct (base)", base))
    inf_models += [("PPO (GAE + critic)", ppo), ("GRPO (no critic)", grpo)]
    if any(m.get("avg_output_tokens") is not None for _, m in inf_models):
        lines += [
            "",
            "## Inference profile (final model, batched vLLM over the test set)",
            "",
            "| Model | Avg output tokens | Avg input tokens | Total gen time (s) | Decode throughput (tok/s) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for label, m in inf_models:
            lines.append(
                f"| {label} | {_fmt(m.get('avg_output_tokens'))} "
                f"| {_fmt(m.get('avg_input_tokens'))} "
                f"| {_fmt(m.get('eval_seconds'))} "
                f"| {_fmt(m.get('decode_throughput_tok_s'))} |"
            )

    lines += [
        "",
        "Notes:",
        "- Accuracy uses the same answer matcher as the training reward.",
        "- Base model (if shown) is run through the same eval harness, untrained.",
        "- Lower peak memory / GPU-hours for GRPO reflects dropping the critic.",
        "- Throughput and val-score are best-effort parses of the veRL console log.",
        "- Inference profile is batched vLLM generation: total gen time is wall-clock "
        "for all test prompts run concurrently, not per-request latency. Avg output "
        "tokens is the conciseness signal.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--hourly-rate", type=float, default=1.50,
                    help="USD per GPU-hour for the cost estimate")
    args = ap.parse_args()

    ppo = collect_run(args.results_dir / "ppo", args.hourly_rate)
    grpo = collect_run(args.results_dir / "grpo", args.hourly_rate)

    # Optional base-model baseline: only present if eval_final.py was run on the
    # untrained model (results/base/eval.json). Enables the 3-way accuracy table.
    base = None
    if (args.results_dir / "base" / "eval.json").exists():
        base = collect_run(args.results_dir / "base", args.hourly_rate)

    report = {"hourly_rate_usd": args.hourly_rate, "ppo": ppo, "grpo": grpo}
    if base is not None:
        report["base"] = base
    (args.results_dir / "comparison_report.json").write_text(json.dumps(report, indent=2))
    md = render_md(ppo, grpo, args.hourly_rate, base=base)
    (args.results_dir / "comparison_report.md").write_text(md)

    print(md)
    print(f"\n[collect] wrote {args.results_dir}/comparison_report.{{json,md}}")


if __name__ == "__main__":
    main()
