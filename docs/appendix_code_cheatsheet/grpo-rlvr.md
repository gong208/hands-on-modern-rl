# C.4 GRPO 与 Reward Model

---

## GRPO Loss

### 一句话记忆

> **同一题采一群答案，组内比个高低当优势；其余照抄 PPO，砍掉 Critic。**

### 伪代码

```
# 第 1 步：同一题让模型生成 G 个回答
completions = [generate(prompt) for _ in range(G)]

# 第 2 步：给每个回答打分
rewards = [reward_fn(c) for c in completions]   # [G]

# 第 3 步：组内归一化（减均值除标准差）→ 当作 advantage
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)

# 第 4 步：套 PPO clipped loss（advantage 来自第 3 步，不是 Critic）
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
policy_loss = -min(surr1, surr2).mean()

# 第 5 步：加 KL 惩罚（拉住，别离参考模型太远）
kl = kl_penalty(log_probs, ref_log_probs)

# 第 6 步：总 loss
loss = policy_loss + kl_coeff * kl
```

### 记忆方法

GRPO = **G**roup **R**elative **P**olicy **O**ptimization。和 PPO 的对比：

|                | PPO                            | GRPO                              |
| -------------- | ------------------------------ | --------------------------------- |
| Advantage 来源 | Critic 预测 $V(s)$ → GAE       | 组内 reward 归一化                |
| 需要几个模型   | 4 个（actor, critic, ref, rm） | 2~3 个（actor, ref, rm/verifier） |
| KL             | 可选                           | 几乎必加                          |
| 采样方式       | 单条 rollout                   | 同 prompt 采 G 条                 |

口诀：**"PPO 砍掉 Critic，换成组内 z-score，其余照抄"**

### Python 实现

```python
import numpy as np

def grpo_advantages(rewards):
    """
    rewards: [num_prompts, G]  每个 prompt 的 G 条回答的 reward
    """
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + 1e-8)

def grpo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """和 PPO clipped loss 完全相同"""
    ratio = np.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def grpo_loss(log_probs, old_log_probs, ref_log_probs,
              rewards, clip_eps=0.2, kl_coeff=0.05):
    """
    log_probs:     [B, G, seq_len]  当前策略
    old_log_probs: [B, G, seq_len]  采样时策略
    ref_log_probs: [B, G, seq_len]  参考策略
    rewards:       [B, G]           组内 reward
    B = num_prompts, G = group_size
    """
    B, G = rewards.shape

    # 1. 组内归一化
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) \
                 / (rewards.std(dim=1, keepdim=True) + 1e-8)
    # [B, G] → [B, G, 1] 以广播到 seq_len 维度
    advantages = advantages.unsqueeze(-1)

    # 2. 序列级 log_prob 求和（每条 completion）
    # 假设 log_probs 已按有效 token 求和: [B, G]
    seq_logp = log_probs.sum(dim=-1)       # [B, G]
    seq_old  = old_log_probs.sum(dim=-1)
    seq_ref  = ref_log_probs.sum(dim=-1)

    # 3. Clipped policy loss
    ratio = torch.exp(seq_logp - seq_old)
    adv = advantages.squeeze(-1)            # [B, G]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # 4. KL 惩罚（k3 估计：r = π_ref / π_θ，样本来自 π_θ）
    log_ratio = seq_ref - seq_logp
    kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()

    return policy_loss + kl_coeff * kl
```

---

## Reward Model（Bradley-Terry 模型）

### 一句话记忆

> **让奖励模型给好回答打分比坏回答高。一行：`-log_sigmoid(r_chosen - r_rejected)`。**

### 伪代码

```
# 第 1 步：奖励模型给两个回答各打一个分
r_w = reward_model(chosen_input)     # 好回答的分数
r_l = reward_model(rejected_input)   # 坏回答的分数

# 第 2 步：希望好分数比坏分数高，过 sigmoid 取负对数
loss = -log(sigmoid(r_w - r_l))
```

### 记忆方法

Bradley-Terry 模型假设人类偏好概率为：

$$P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$$

训练目标就是最大化这个概率的对数，等价于最小化 `-log_sigmoid(diff)`。

口诀：**"RM 训练就是 pairwise 交叉熵"**

### Python 实现

```python
def log_sigmoid(x):
    return -np.logaddexp(0, -x)

def reward_model_loss(r_chosen, r_rejected):
    """r_chosen, r_rejected: [B]"""
    return -log_sigmoid(r_chosen - r_rejected).mean()
```

### PyTorch 实现

```python
def reward_model_loss(r_chosen, r_rejected):
    """
    r_chosen:  [B]  reward model 对 chosen 的打分
    r_rejected: [B]  reward model 对 rejected 的打分
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

---

## 面试追问：DPO 和 RLHF-PPO 的关系

面试官常问"DPO 相比 PPO 的优劣"，准备这个对比表：

| 维度                 | PPO-RLHF             | DPO                  |
| -------------------- | -------------------- | -------------------- |
| 需要 Reward Model    | 是                   | 否（隐式学习）       |
| 需要 Critic          | 是                   | 否                   |
| 需要 Reference Model | 可选                 | 必须                 |
| 在线/离线            | 在线（需要采样）     | 离线（只用偏好数据） |
| 训练成本             | 高（4 个模型）       | 低（2 个模型）       |
| 奖励黑客风险         | 有（RM 可被钻空子）  | 较低（无显式 RM）    |
| 理论最优性           | 更强（可以持续探索） | 受限于离线数据质量   |
| 适用场景             | 大规模在线训练       | 偏好数据充足的场景   |

---

## 易错点

| 易错                           | 说明                                                                          |
| ------------------------------ | ----------------------------------------------------------------------------- |
| GRPO 的 advantage 是组内归一化 | 不是全局归一化，是**同一个 prompt** 的 G 条回答之间比较                       |
| GRPO 没有 value loss           | 没有 Critic，所以没有 value loss，这是和 PPO 的核心区别                       |
| Reward Model 要 detach         | 训练 RM 时 chosen/rejected 的 reward 都要参与梯度，但训练 policy 时 RM 要冻结 |
| GRPO 的 KL 是对每条序列的      | 不是 token 级别，通常是对整条 completion 的 log_prob 求和后再算 KL            |
| DPO 隐式学到了 RM              | DPO 的 `log_ratio_w - log_ratio_l` 本质上就是隐式 reward 差值                 |
| G 的大小                       | 通常 G=4~16，太小 advantage 估计噪声大，太大采样成本高                        |
| RLVR 场景                      | reward 来自规则验证器（如代码执行、数学答案检查），不是 RM 打分               |
