# C.1 SFT Loss 与 KL 散度

## SFT Loss（自回归交叉熵）

### 一句话记忆

> **input 右移一位当 target，只在 answer 部分（`label != -100`）算交叉熵。**

### 伪代码

```
logits = model(input_ids)          # [B, seq_len, vocab_size]
shift_logits = logits[:, :-1, :]   # 去掉最后一个位置的预测
shift_labels = input_ids[:, 1:]    # 去掉第一个 token

loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

### 为什么 shift right？

自回归模型在每个位置 $t$ 预测 $t+1$ 的 token。所以 logits 的第 $t$ 个位置对应 labels 的第 $t+1$ 个位置。"左砍 logits 尾，右砍 labels 头"。

### Python 实现

```python
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [seq_len, vocab_size]
    labels:  [seq_len]  (已 shift)
    """
    shift_logits = logits[:-1]   # 去尾
    shift_labels = labels[1:]    # 去头

    probs = softmax(shift_logits, axis=-1)
    total, count = 0.0, 0

    for t in range(len(shift_labels)):
        if shift_labels[t] == ignore_index:
            continue
        total += -np.log(probs[t, shift_labels[t]] + 1e-12)
        count += 1

    return total / max(count, 1)
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [B, seq_len, vocab_size]
    labels: [B, seq_len]  (原始 input_ids，函数内部做 shift)
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

## KL 散度估计

### 一句话记忆

> **KL(p || q) = E_p[log p - log q]。两个写法：有偏用 `log_ratio`，无偏用 `ratio - 1 - log_ratio`。**

面试常考：PPO 里怎么算 KL？GRPO 里怎么算 KL？两种估计有何区别？

### 伪代码

```
# 方法一：有偏估计（简单，PPO 常用）
kl = (log_prob - ref_log_prob).mean()

# 方法二：无偏估计（GRPO /trl 常用）
ratio = exp(log_prob - ref_log_prob)
kl = (ratio - 1 - (log_prob - ref_log_prob)).mean()
```

### Python 实现

```python
import numpy as np

def kl_biased(log_p, log_q):
    """有偏估计：E_p[log p - log q]，简单但可能为负"""
    return np.mean(log_p - log_q)

def kl_unbiased(log_p, log_q):
    """无偏估计：E_p[exp(log p - log q) - 1 - (log p - log q)]"""
    log_ratio = log_p - log_q
    return np.mean(np.exp(log_ratio) - 1 - log_ratio)
```

### PyTorch 实现

```python
import torch

def kl_penalty(log_probs, ref_log_probs, mode="unbiased"):
    """
    log_probs:     [B, seq_len]  当前策略的 log 概率
    ref_log_probs: [B, seq_len]  参考策略的 log 概率
    """
    if mode == "biased":
        return (log_probs - ref_log_probs).mean()

    # 无偏估计（trl / GRPO 默认）
    log_ratio = log_probs - ref_log_probs
    return (torch.exp(log_ratio) - 1 - log_ratio).mean()
```

### 两种估计的区别

| 估计方法 | 公式 | 特点 |
|----------|------|------|
| 有偏 | $\mathbb{E}_p[\log \frac{p}{q}]$ | 简单，但样本少时可能为负 |
| 无偏 | $\mathbb{E}_p[\frac{p}{q} - 1 - \log \frac{p}{q}]$ | 保证 $\geq 0$，GRPO 默认 |

---

## 易错点

| 易错 | 说明 |
|------|------|
| shift 方向反了 | logits 砍**尾**，labels 砍**头**。口诀："预测看左边，目标看右边" |
| 忘了 `ignore_index` | prompt 部分的 token 不参与 loss，设为 `-100` |
| KL 符号搞反 | KL(p \|\| q) 里 p 是当前策略、q 是参考策略，写反了变成负数 |
| softmax 溢出 | 先减 `max(x)` 再 `exp`，面试手写必加 |
| `contiguous()` | PyTorch 里 slice 后 view 会报错，加 `.contiguous()` |
