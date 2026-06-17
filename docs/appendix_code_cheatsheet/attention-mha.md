# C.7 Self-Attention、MHA、MQA 与 GQA

严格来说这不是 RL 算法，但它是大模型岗面试中**出现频率前三**的手写代码题。RL 岗面试也经常作为前置知识考查。

---

## Scaled Dot-Product Attention

### 一句话记忆

> **Q 和 K 点积打分，除根号、遮未来、softmax、再乘 V。**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 伪代码

```
# 第 1 步：Q 和 K 点积 = 每个 Q 看每个 K 的相似度，除根号 d_k 防止数值过大
scores = Q @ K^T / sqrt(d_k)

# 第 2 步：加 mask，把"未来"位置压成 -inf（语言模型只能看左边）
scores = scores + mask

# 第 3 步：softmax 转成权重（加起来等于 1）
attn_weights = softmax(scores, dim=-1)

# 第 4 步：权重乘 V，得到加权后的输出
output = attn_weights @ V
```

### 记忆方法

三步走：

1. **打分**：Q 和 K 的点积衡量相似度，除 $\sqrt{d_k}$ 防止点积过大导致 softmax 饱和
2. **遮掩**：causal mask 把"未来"位置设为 $-\infty$（语言模型只能看左边）
3. **加权**：softmax 后的权重乘 V，得到加权表示

### Python 实现

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [seq_len, d_k]
    K: [seq_len, d_k]
    V: [seq_len, d_v]
    mask: [seq_len, seq_len]  0=保留, -inf=遮掩
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    # softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return attn_weights @ V
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [B, heads, seq_len, d_k]
    K: [B, heads, seq_len, d_k]
    V: [B, heads, seq_len, d_v]
    mask: [1, 1, seq_len, seq_len]  或 [B, 1, 1, seq_len]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)

def causal_mask(seq_len):
    """生成因果遮掩：上三角为 0（遮掩），下三角为 1（保留）"""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
```

---

## Multi-Head Attention (MHA)

### 一句话记忆

> **总维度切成 h 份，每份单独做一次 attention，最后拼回去过线性层。**

### 伪代码

```
# 第 1 步：x 过三个线性层，得到 Q、K、V（每个仍是 [B, seq, d_model]）
Q = x @ W_Q
K = x @ W_K
V = x @ W_V

# 第 2 步：把最后一维 d_model 切成 h 个头
#   [B, seq, d_model] → [B, heads, seq, d_k]
Q = Q.view(B, seq, heads, d_k).transpose(1, 2)
K = K.view(B, seq, heads, d_k).transpose(1, 2)
V = V.view(B, seq, heads, d_k).transpose(1, 2)

# 第 3 步：每个头独立做 attention
attn_out = scaled_dot_product_attention(Q, K, V, mask)

# 第 4 步：把头拼回 d_model，再过输出线性层
attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq, d_model)
output = attn_out @ W_O
```

### 记忆方法

一个头的 attention 只能看一种"关系模式"。多头让模型同时关注不同位置的 不同表示子空间。

维度变换口诀：**"view 切头，transpose 换位，attention 计算，transpose 回来，view 合头"**

### PyTorch 实现

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, seq_len, d_model = x.shape

        # 线性投影 + 切头
        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_out = scaled_dot_product_attention(Q, K, V, mask)

        # 合头 + 输出投影
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        return self.W_O(attn_out)
```

---

## MQA 与 GQA

### 对比速查

| 变体 | Q 的头数 | K/V 的头数    | K/V 参数量             | 代表模型           |
| ---- | -------- | ------------- | ---------------------- | ------------------ |
| MHA  | h        | h             | $3 \times d_{model}^2$ | GPT-2、BERT        |
| MQA  | h        | **1**         | 大幅减少               | PaLM、StarCoder    |
| GQA  | h        | **g** (g < h) | 折中                   | LLaMA 2/3、Mistral |

### 一句话记忆

- **MQA**：所有 Q 头共用**同一组** K/V。最省显存，但可能变笨。
- **GQA**：Q 头分几组，组内共用 K/V。省显存又不至于太笨。

### PyTorch 实现（GQA）

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        """
        n_heads: Q 的头数 (如 32)
        n_kv_heads: K/V 的头数 (如 8)
        n_heads 必须能被 n_kv_heads 整除
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # 每组几个 Q 头
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_O = nn.Linear(n_heads * self.d_k, d_model)

    def forward(self, x, mask=None):
        B, seq_len, _ = x.shape

        # Q: [B, seq, n_heads * d_k] → [B, n_heads, seq, d_k]
        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # K/V: [B, seq, n_kv_heads * d_k] → [B, n_kv_heads, seq, d_k]
        K = self.W_K(x).view(B, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 扩展 K/V 以匹配 Q 的头数: [B, n_kv_heads, seq, d_k] → [B, n_heads, seq, d_k]
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.W_O(attn_out)
```

---

## 面试追问：计算复杂度

|                | 复杂度            | 说明                       |
| -------------- | ----------------- | -------------------------- |
| Self-Attention | $O(n^2 \cdot d)$  | $n$ 是序列长度，$d$ 是维度 |
| 线性投影       | $O(n \cdot d^2)$  | 每个token 过线性层         |
| 总计（MHA）    | $O(n^2 d + nd^2)$ | 长序列时 $n^2$ 项主导      |

---

## 易错点

| 易错                                    | 说明                                                             |
| --------------------------------------- | ---------------------------------------------------------------- |
| 除 $\sqrt{d_k}$ 不是 $\sqrt{d_{model}}$ | 是每个头的维度，不是总维度                                       |
| causal mask 方向                        | `tril` 生成下三角 = 保留，上三角 = 遮掩（未来）                  |
| view 前 contiguous                      | transpose 后内存不连续，必须先 `.contiguous()` 再 view           |
| GQA 的 repeat_interleave                | 不是 repeat，是 `repeat_interleave`，保证相邻 Q 头共享同一组 K/V |
| MQA 是 GQA 的特例                       | 当 `n_kv_heads=1` 时 GQA 退化为 MQA                              |
