# C.6 Top-k / Top-p Sampling 与 Temperature

解码策略是 LLM 面试的常考题，和 RL 直接相关（RLHF 训练后的模型怎么采样、temperature 如何影响策略分布）。

---

## Temperature

### 一句话记忆

> **logits 除以 T 再 softmax。T 大更随机，T 小更确定。**

### 伪代码

```
# 第 1 步：logits 除以 T
#   T 大 → 数之间差距变小 → 概率更平均
#   T 小 → 数之间差距变大 → 概率更尖
scaled_logits = logits / temperature

# 第 2 步：softmax 转成概率
probs = softmax(scaled_logits)
```

### 记忆方法

- $T \to 0$：趋向 argmax（贪婪），相当于确定性策略
- $T = 1$：原始分布
- $T \to \infty$：趋向均匀分布，相当于随机策略

RL 视角：temperature 就是策略的探索程度。

### PyTorch 实现

```python
def sample_with_temperature(logits, temperature=1.0):
    if temperature < 1e-8:
        return logits.argmax(dim=-1)  # 贪婪
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-k Sampling

### 一句话记忆

> **只留概率最高的 k 个，其他全砍掉（设 -inf），再采样。**

### 伪代码

```
# 第 1 步：找出 logits 最大的 k 个值，记下阈值
top_k_values, top_k_indices = topk(logits, k)

# 第 2 步：低于阈值的全部设成 -inf（softmax 后变 0）
logits[not in top_k] = -inf

# 第 3 步：softmax 重新归一化，从剩下 k 个里采样
probs = softmax(logits)
sample from probs
```

### Python 实现

```python
import numpy as np

def top_k_filtering(logits, k):
    """
    logits: [vocab_size]
    返回: 过滤后的 logits（非 top-k 位置为 -inf）
    """
    if k >= len(logits):
        return logits
    threshold = np.sort(logits)[-k]  # 第 k 大的值
    logits_filtered = np.where(logits >= threshold, logits, -np.inf)
    return logits_filtered
```

### PyTorch 实现

```python
import torch

def top_k_filtering(logits, k):
    """
    logits: [B, vocab_size] 或 [vocab_size]
    """
    if k <= 0:
        return logits
    top_k = min(k, logits.size(-1))
    # 找到第 k 大的值作为阈值
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float('-inf'))

def top_k_sample(logits, k, temperature=1.0):
    logits = top_k_filtering(logits, k)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-p (Nucleus) Sampling

### 一句话记忆

> **按概率从大到小排，累加到 p 就停——只从这群高概率 token 里采样。**

### 伪代码

```
# 第 1 步：logits 从大到小排序
sorted_logits = sort_desc(logits)

# 第 2 步：转概率，再算累积概率（从大到小累加）
sorted_probs = softmax(sorted_logits)
cumulative_probs = cumsum(sorted_probs)

# 第 3 步：累积值超过 p 之后的所有位置，设成 -inf
#   注意：要减去当前 prob 再比，保证至少留一个
cutoff_mask = cumulative_probs - sorted_probs > p
sorted_logits[cutoff_mask] = -inf

# 第 4 步：还原原顺序，softmax，采样
```

### 记忆方法

Top-k 是固定数量，Top-p 是固定概率质量。Top-p 更灵活：确定性强时选几个 token 就够了，不确定时可能需要很多 token 才能凑够 p。

面试常问区别：

|          | Top-k         | Top-p                    |
| -------- | ------------- | ------------------------ |
| 筛选依据 | 固定保留 k 个 | 保留概率质量前 p         |
| 适应性   | 不随分布变化  | 自动适应分布尖锐程度     |
| 极端情况 | k=1 → 贪婪    | p=0 → 贪婪，p=1 → 不限制 |

### Python 实现

```python
import numpy as np

def top_p_filtering(logits, p):
    """
    logits: [vocab_size]
    """
    sorted_indices = np.argsort(logits)[::-1]  # 降序
    sorted_logits = logits[sorted_indices]
    sorted_probs = np.exp(sorted_logits - sorted_logits.max())
    sorted_probs = sorted_probs / sorted_probs.sum()
    cumulative_probs = np.cumsum(sorted_probs)

    # 找到累积概率超过 p 的位置（保留至少一个 token）
    cutoff = cumulative_probs - sorted_probs > p
    sorted_logits[cutoff] = -np.inf

    # 还原顺序
    result = np.empty_like(logits)
    result[sorted_indices] = sorted_logits
    return result
```

### PyTorch 实现

```python
import torch

def top_p_filtering(logits, p):
    """
    logits: [B, vocab_size]
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 移除累积概率超过 p 的 token（保留至少一个）
    sorted_mask = (cumulative_probs - sorted_probs) > p
    sorted_logits[sorted_mask] = float('-inf')

    # 还原原始顺序
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_p_sample(logits, p, temperature=1.0):
    logits = top_p_filtering(logits, p)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 实际组合用法

工程中通常 Top-k + Top-p + Temperature 一起用：

```python
def generate_sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    # 1. Temperature 缩放
    logits = logits / max(temperature, 1e-8)
    # 2. Top-k 过滤
    logits = top_k_filtering(logits, top_k)
    # 3. Top-p 过滤
    logits = top_p_filtering(logits, top_p)
    # 4. 采样
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 易错点

| 易错                 | 说明                                                                         |
| -------------------- | ---------------------------------------------------------------------------- |
| Top-p 的 cumsum 方向 | 必须**降序排列**后再 cumsum，升序没有意义                                    |
| Top-p 保留至少一个   | `cumsum - current_prob > p` 而不是 `cumsum > p`，否则第一个 token 可能被误杀 |
| Top-k 的阈值         | 用 `topk().values[..., -1]` 取第 k 大的值，不是 sort 后取 index              |
| 还原顺序             | Top-p 排序后要 scatter 回原位，忘了还原会导致采样错乱                        |
| Temperature=0        | 要特殊处理为 argmax，不能真的除以 0                                          |
