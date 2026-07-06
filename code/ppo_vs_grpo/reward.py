"""Shared reward for the PPO-vs-GRPO comparison.

Both the PPO and the GRPO runs use this exact reward so that the reward signal
is never a confounding variable. Rather than copy the logic, we re-export
``compute_score`` from the chapter 8 PPO project, keeping a single source of
truth:

    code/chapter08_rlhf/verl_gsm8k/gsm8k_reward.py

``compute_score`` returns ``{"score", "accuracy", "format"}`` where ``score`` is
1.0 iff the extracted answer matches the ground truth (binary correctness) plus
a format flag. This matches veRL's ``custom_reward_function`` interface
``(data_source, solution_str, ground_truth, extra_info=None, **kwargs)``.
"""

import sys
from pathlib import Path

# Resolve the chapter 8 reward regardless of where the repo is mounted
# (e.g. /workspace inside the container). This file lives at
# <repo>/code/ppo_vs_grpo/reward.py, so the sibling project is two parents up.
_CH08_DIR = Path(__file__).resolve().parents[1] / "chapter08_rlhf" / "verl_gsm8k"
if str(_CH08_DIR) not in sys.path:
    sys.path.insert(0, str(_CH08_DIR))

from gsm8k_reward import check_answer, compute_score, extract_answer  # noqa: E402,F401

__all__ = ["compute_score", "extract_answer", "check_answer"]
