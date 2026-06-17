# C.8 DAPO

DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）是 2025 年字节跳动提出的 GRPO 改进，面试中出现频率快速上升。

---

## DAPO vs GRPO：三个改进

| 改进     | GRPO                             | DAPO                                   |
| -------- | -------------------------------- | -------------------------------------- |
| 裁剪策略 | 对称裁剪 `clip(ratio, 1-ε, 1+ε)` | **解耦裁剪**：正/负 advantage 分开裁剪 |
| 采样策略 | 固定 prompt                      | **动态采样**：过滤全对/全错的 prompt   |
| 超长惩罚 | 二元（超长=0 分）                | **渐进惩罚**：超长越多扣越多           |

---

## 解耦裁剪（Decoupled Clip）

### 一句话记忆

> **好的不贪心、坏的不报复：正优势只裁上界，负优势只裁下界。**

### 伪代码

```
# 第 1 步：算新旧策略的比值
ratio = exp(new_logp - old_logp)

# 第 2 步：正 advantage —— 只裁上界（不让 ratio 飙太高）
pos_surr = min(ratio, 1 + eps) * advantage    # advantage > 0

# 第 3 步：负 advantage —— 只裁下界（允许 ratio 回弹）
neg_surr = max(ratio, 1 - eps) * advantage    # advantage < 0

# 第 4 步：合并取平均
loss = -mean(pos_surr + neg_surr)
```

### 记忆方法

画图对比：

```
GRPO (对称裁剪):
  advantage > 0:  min(ratio, 1+ε) * A    ← 裁上界
  advantage < 0:  max(ratio, 1-ε) * A    ← 裁下界
  → 两边都裁，比较保守

DAPO (解耦裁剪):
  advantage > 0:  min(ratio, 1+ε_high) * A   ← 裁上界，ε_high 可以更大
  advantage < 0:  max(ratio, 1-ε_low)  * A   ← 裁下界，ε_low 可以更小
  → 允许独立调两个方向的探索力度
```

口诀：**"正优裁上防贪心，负优裁下防报复，两个 ε 各自调"**

### Python 实现

```python
import numpy as np

def dapo_policy_loss(new_logp, old_logp, advantages,
                     clip_high=0.28, clip_low=0.28):
    """
    new_logp: [T]
    old_logp: [T]
    advantages: [T]
    clip_high: 正 advantage 的上界裁剪
    clip_low:  负 advantage 的下界裁剪
    """
    ratio = np.exp(new_logp - old_logp)

    pos_mask = advantages >= 0
    neg_mask = ~pos_mask

    loss = np.zeros_like(advantages)

    # 正 advantage: 只裁上界
    if pos_mask.any():
        clipped_ratio = np.minimum(ratio[pos_mask], 1 + clip_high)
        loss[pos_mask] = -(clipped_ratio * advantages[pos_mask])

    # 负 advantage: 只裁下界
    if neg_mask.any():
        clipped_ratio = np.maximum(ratio[neg_mask], 1 - clip_low)
        loss[neg_mask] = -(clipped_ratio * advantages[neg_mask])

    return loss.mean()
```

### PyTorch 实现

```python
import torch

def dapo_policy_loss(new_logps, old_logps, advantages,
                     clip_high=0.28, clip_low=0.28):
    """
    new_logps:  [B, seq_len]
    old_logps:  [B, seq_len]
    advantages: [B, seq_len]
    """
    ratio = torch.exp(new_logps - old_logps)

    pos_mask = advantages >= 0
    neg_mask = ~pos_mask

    loss = torch.zeros_like(advantages)

    # 正 advantage: min(ratio, 1 + clip_high) * advantage
    if pos_mask.any():
        clipped = torch.clamp(ratio[pos_mask], max=1 + clip_high)
        loss[pos_mask] = -(clipped * advantages[pos_mask])

    # 负 advantage: max(ratio, 1 - clip_low) * advantage
    if neg_mask.any():
        clipped = torch.clamp(ratio[neg_mask], min=1 - clip_low)
        loss[neg_mask] = -(clipped * advantages[neg_mask])

    return loss.mean()
```

---

## 动态采样（Dynamic Sampling）

### 一句话记忆

> **一组答案全对或全错 → 没区分度 → 跳过这个题。**

### 伪代码

```
# 遍历每个题（prompt）
for each prompt:
    # 题下采的 G 条回答都打分
    rewards = [get_reward(completion) for completion in group]

    # 全部分数一样（全对或全错）→ 没区分度，跳过
    if all rewards are the same:
        skip this prompt
```

### PyTorch 实现

```python
def dynamic_sampling_filter(rewards):
    """
    rewards: [B, G]  B 个 prompt，每个 G 条回答的 reward
    返回: bool mask [B]，True = 保留
    """
    reward_std = rewards.std(dim=1)  # 每组的 reward 标准差
    return reward_std > 1e-6         # 有区分度才保留
```

### 记忆方法

GRPO 的 advantage 是组内 z-score 归一化。如果全组 reward 一样，std=0，advantage 全是 NaN/0。DAPO 直接在数据层面过滤掉这些无效样本，而不是等到 loss 计算时才发现问题。

---

## 超长惩罚（Overlong Reward Shaping）

### 一句话记忆

> **回答超长不全扣光，按超出多少慢慢扣。**

### 伪代码

```
# 第 1 步：超过最大长度才扣分
if response_length > max_length:
    # 第 2 步：超得越多扣得越多（按超出比例）
    penalty_ratio = (response_length - max_length) / max_length
    # 第 3 步：从原 reward 里减掉
    reward = reward - penalty_weight * penalty_ratio
```

### Python 实现

```python
def overlong_reward_shaping(reward, response_length,
                            max_length, penalty_weight=0.1):
    if response_length <= max_length:
        return reward
    penalty = penalty_weight * (response_length - max_length) / max_length
    return reward - penalty
```

### 记忆方法

对比 GRPO 的做法：

- GRPO：超长 → reward = 0（二元，突变）
- DAPO：超长 → reward 线性递减（平滑，有梯度信号）

RL 视角：二元奖励在边界处没有梯度，策略不知道"短一点就好了"。线性惩罚给出方向信号。

---

## DAPO 总 Loss

```
# 1. 组内归一化（和 GRPO 相同）
advantages = (rewards - mean) / (std + eps)

# 2. 动态采样过滤
valid_mask = dynamic_sampling_filter(rewards)

# 3. 解耦裁剪 policy loss
policy_loss = dapo_policy_loss(new_logp, old_logp, advantages, clip_high, clip_low)

# 4. KL 惩罚
kl = kl_penalty(log_probs, ref_log_probs)

# 5. 总 loss
loss = policy_loss[valid_mask].mean() + kl_coeff * kl
```

---

## GRPO vs DAPO 完整对比

| 维度       | GRPO                     | DAPO                             |
| ---------- | ------------------------ | -------------------------------- |
| 裁剪       | 对称 `clip(r, 1-ε, 1+ε)` | 解耦，正/负 advantage 各自一个 ε |
| 无效数据   | 不处理（std=0 时 NaN）   | 动态采样过滤                     |
| 超长奖励   | 二元（0/1）              | 渐进线性惩罚                     |
| 探索灵活性 | 固定                     | 正方向可以更激进，负方向更保守   |
| 代表工作   | DeepSeek-R1              | ByteDance/清华 DAPO              |

---

## 易错点

| 易错                               | 说明                                                   |
| ---------------------------------- | ------------------------------------------------------ |
| 解耦裁剪不是取消裁剪               | 仍然有裁剪，只是正/负方向独立，ε 可以不同              |
| 动态采样的判断条件                 | 不是"reward 低于阈值"，而是"组内 reward **方差为零**"  |
| 超长惩罚是线性不是指数             | 简单的 `(len - max_len) / max_len`，不需要更复杂的形式 |
| DAPO 的 advantage 仍然是组内归一化 | 这部分和 GRPO 完全一样                                 |
| clip_high 和 clip_low 可以不同     | 面试追问时说"可以根据任务调整两个方向的探索力度"       |
