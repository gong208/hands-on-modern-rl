# C.2 PPO 策略损失与 GAE

PPO 是大模型 RL 面试中考查频率最高的算法。面试官通常会要求你写出 **clipped policy loss**，可能追问 value loss 和 GAE。

---

## GAE（广义优势估计）

### 一句话记忆

> **从最后一步往回扫：当下奖励叠加未来的优势。**

GAE 是 PPO 的前置知识，面试常单独问。

### 伪代码

```
# 第 1 步：算 TD 误差 δ_t
#   = 当前奖励 + 折扣后的下一步估值 - 当前估值
delta_t = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t

# 第 2 步：从后往前叠 —— 这一步的 δ 加上折扣后的"未来优势"
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_{t+1}

# 第 3 步：return = advantage + 当前估值（给 critic 学）
return_t = advantage_t + value_t
```

### 记忆方法

把 GAE 想成一个**指数加权移动平均**：

- $\lambda = 0$：退化为单步 TD（只看 $\delta_t$），方差低、偏差高
- $\lambda = 1$：退化为蒙特卡洛回报（累加所有 $\delta$），方差高、偏差低
- 面试口诀：**"lambda 越大越敢看未来"**

### Python 实现

```python
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T+1]  (最后一个是 bootstrap value)
    dones:   [T]
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0.0

    for t in reversed(range(T)):
        # TD error
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        # 逆向累积
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values[:T]
    return advantages, returns
```

### PyTorch 实现

```python
import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [B, T]
    values:  [B, T+1]
    dones:   [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B)

    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
        last_adv = delta + gamma * lam * (1 - dones[:, t]) * last_adv
        advantages[:, t] = last_adv

    returns = advantages + values[:, :T]
    return advantages, returns
```

---

## PPO Clipped Policy Loss

### 一句话记忆

> **新策略 ÷ 旧策略 = 比值；比值 × 优势 = 目标；比值超范围就截断，取小的那个。**

$$L^{CLIP} = -\min\big(r_t(\theta) \cdot A_t,\;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\big)$$

### 伪代码

```
# 第 1 步：算新旧策略的比值（用 log 相减再 exp，数值更稳）
ratio = exp(new_log_prob - old_log_prob)

# 第 2 步：算两个 surrogate —— 一个不截断，一个截断
surr1 = ratio * advantage
surr2 = clip(ratio, 1-eps, 1+eps) * advantage

# 第 3 步：取较小的（更保守），加负号变最小化
loss = -min(surr1, surr2).mean()
```

### 记忆方法

画一条数轴，ratio 为横轴，loss 为纵轴：

- advantage > 0 时：策略变好要鼓励，但 ratio 超过 $1+\epsilon$ 后截断，**不贪心**
- advantage < 0 时：策略变差要惩罚，但 ratio 低于 $1-\epsilon$ 后截断，**不报复**

口诀：**"正优截上，负优截下，min 取保守"**

### Python 实现

```python
import numpy as np

def ppo_policy_loss(new_logp, old_logp, advantages, clip_eps=0.2):
    """
    new_logp:   [T]  当前策略的 log 概率
    old_logp:   [T]  采样时策略的 log 概率
    advantages: [T]
    """
    ratio = np.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -np.minimum(surr1, surr2).mean()
    return loss
```

### PyTorch 实现

```python
import torch

def ppo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """
    new_logps:   [B, T]
    old_logps:   [B, T]
    advantages:  [B, T]
    """
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()
    return loss
```

---

## PPO Value Loss

### 一句话记忆

> **猜的值离目标多远，平方一下取平均；可选：预测别离旧预测太远。**

### 伪代码

```
# 第 1 步：critic 给出当前预测
value_pred = critic(state)

# 第 2 步：把预测的变化幅度截断，得到一个"保守版"预测
value_clipped = old_values + clip(value_pred - old_values, -eps, eps)

# 第 3 步：两个版本都算 MSE，取较大的（更保守）
loss1 = (value_pred - returns)^2
loss2 = (value_clipped - returns)^2
loss = max(loss1, loss2).mean()
```

### PyTorch 实现

```python
def ppo_value_loss(values, old_values, returns, clip_eps=0.2):
    loss1 = (values - returns) ** 2
    values_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    loss2 = (values_clipped - returns) ** 2
    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## PPO 总 Loss

```
total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
```

面试追问时，完整的 PPO loss 三件套：

| 组成                  | 作用        | 系数典型值      |
| --------------------- | ----------- | --------------- |
| policy loss (clipped) | 更新策略    | 权重 1          |
| value loss (MSE)      | 更新 Critic | `vf_coef=0.5`   |
| entropy bonus         | 鼓励探索    | `ent_coef=0.01` |

---

## 易错点

| 易错               | 说明                                                                                   |
| ------------------ | -------------------------------------------------------------------------------------- |
| ratio 用除法       | 应该用 `exp(logp_new - logp_old)`，数值更稳定                                          |
| advantage 没归一化 | 实际工程中 advantage 通常做 batch 内归一化                                             |
| min 还是 max       | 对 loss 取 min（保守），对 value loss 取 max（也是保守方向）                           |
| 忘了 stop gradient | `old_log_probs` 和 `old_values` 要 `.detach()`                                         |
| GAE 的 done mask   | `done=1` 时截断递推：`gamma * lambda * (1-done) * next_adv`                            |
| value 的 bootstrap | values 长度是 T+1，最后一个位置是 bootstrap value                                      |
| entropy 符号       | `- entropy_coeff * entropy`（entropy 本身是正的，前面要加负号让 loss 变小 = 鼓励高熵） |
