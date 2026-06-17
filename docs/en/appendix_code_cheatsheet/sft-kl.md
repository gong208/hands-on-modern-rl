---
title: C.1 SFT and KL
---

# C.1 SFT and KL

## SFT Loss (Autoregressive Cross-Entropy)

### One-Line Memory

> Read the previous word, guess the next. The question doesn't count; only the answer does.

### Pseudocode

```
# Step 1: run the model on the full sequence; each position predicts its "next word"
logits = model(input_ids)

# Step 2: shift to align — position t predicts the real word at position t+1
#   drop the last logits slot: there is no "next word" after the end
#   drop the first label: nobody predicts the very first word
shift_logits = logits[:, :-1, :]   # cut the tail
shift_labels = input_ids[:, 1:]    # cut the head

# Step 3: cross-entropy between prediction and truth; question tokens (=-100) are skipped
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

### Why Shift Right?

An autoregressive model predicts the token at position $t+1$ from the prefix up to position $t$. Therefore, the logits at index $t$ correspond to the labels at index $t+1$.

A simple mnemonic: "cut the tail of logits, cut the head of labels."

### Python (NumPy) Implementation

```python
import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [seq_len, vocab_size]
    labels: [seq_len] (unshifted; we shift inside)
    """
    shift_logits = logits[:-1]  # drop tail
    shift_labels = labels[1:]  # drop head

    probs = softmax(shift_logits, axis=-1)
    total, count = 0.0, 0

    for t in range(len(shift_labels)):
        if shift_labels[t] == ignore_index:
            continue
        total += -np.log(probs[t, shift_labels[t]] + 1e-12)
        count += 1

    return total / max(count, 1)
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F


def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [B, seq_len, vocab_size]
    labels: [B, seq_len] (typically the original input_ids; we shift inside)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss
```

---

## KL Divergence Estimates

### One-Line Memory

> Measure how far apart two models are. The simple estimator can go negative; the safe one never does.

Interview-style questions:

- How do you compute KL in PPO?
- How do you compute KL in GRPO?
- What is the difference between these estimates?

### Pseudocode

```
# Method 1: k1 (simple version, common in PPO)
# Idea: average the difference (current - reference)
# Downside: with few samples the estimate can go negative
kl = (log_prob - ref_log_prob).mean()

# Method 2: k3 (safe version, default in GRPO / trl)
# Step 1: compute log(reference / current) — note the direction is flipped
log_ratio = ref_log_prob - log_prob      # ratio = q/p
# Step 2: apply the formula exp(x) - 1 - x; this expression is guaranteed >= 0
kl = (exp(log_ratio) - 1 - log_ratio).mean()
```

### Python (NumPy) Implementation

```python
import numpy as np


def kl_k1(log_p, log_q):
    """k1: E_p[log p - log q]. Unbiased but high variance; can be negative with few samples."""
    return np.mean(log_p - log_q)


def kl_k3(log_p, log_q):
    """k3: E_p[exp(log q - log p) - 1 - (log q - log p)]. Unbiased and always nonnegative."""
    log_ratio = log_q - log_p
    return np.mean(np.exp(log_ratio) - 1 - log_ratio)
```

### PyTorch Implementation

```python
import torch


def kl_penalty(log_probs, ref_log_probs, mode="k3"):
    """
    log_probs:     [B, seq_len] log-probabilities under the current policy
    ref_log_probs: [B, seq_len] log-probabilities under the reference policy
    """
    if mode == "k1":
        # k1: unbiased but high variance; can be negative with few samples
        return (log_probs - ref_log_probs).mean()

    # k3: unbiased and always nonnegative (default in trl / GRPO)
    log_ratio = ref_log_probs - log_probs     # ratio = q/p
    return (torch.exp(log_ratio) - 1 - log_ratio).mean()
```

### What Is the Difference?

Samples come from the current policy $p$; the target is $\text{KL}(p \| q)$ with $q$ the reference policy:

| Estimator | Formula                                            | Notes                                                          |
| --------- | -------------------------------------------------- | -------------------------------------------------------------- |
| k1        | $\mathbb{E}_p[\log \frac{p}{q}]$                   | unbiased, simple, but may become negative with limited samples |
| k3        | $\mathbb{E}_p[\frac{q}{p} - 1 - \log \frac{q}{p}]$ | unbiased and always $\ge 0$, default in GRPO                   |

::: warning Pitfall
In k3 the ratio must be $q/p$ (ref/current), not $p/q$. Flipping the sign makes the expectation $\chi^2(p\|q) - \text{KL}(p\|q)$ — still nonnegative, but no longer the KL, and inconsistent with the Chapter 9 formula.
:::

---

## Common Pitfalls

| Pitfall                  | Explanation                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Shift direction is wrong | Cut the **tail** of logits and the **head** of labels.                                                             |
| Forgot `ignore_index`    | Prompt tokens should not contribute to the loss; they are usually masked with `-100`.                              |
| KL arguments swapped     | In $\mathrm{KL}(p \| q)$, $p$ is the current policy and $q$ is the reference policy. Swapping them flips the sign. |
| Softmax overflow         | Subtract `max(x)` before `exp`. This is expected in interviews.                                                    |
| Missing `.contiguous()`  | In PyTorch, slicing can create non-contiguous tensors; `view` then fails unless you call `.contiguous()`.          |
