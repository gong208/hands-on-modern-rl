# C.3 DPO 及其变体

DPO Loss 是后训练岗位面试中考查频率**最高**的手写代码题。几乎每场面试都会考。

---

## DPO Loss

### 一句话记忆

> **"chosen 的 logit 减 rejected 的 logit，再减去 reference 同样做一遭"。四个 log_prob，两个相减，再减 reference 的同样相减。**

$$\mathcal{L}_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\big(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big)\Big)\Big]$$

### 伪代码

```
pi_chosen  = log_pi_theta(y_w | x)
pi_rejected = log_pi_theta(y_l | x)
ref_chosen  = log_pi_ref(y_w | x)
ref_rejected = log_pi_ref(y_l | x)

log_ratio_w = pi_chosen  - ref_chosen       # 当前 vs 参考，chosen
log_ratio_l = pi_rejected - ref_rejected     # 当前 vs 参考，rejected

loss = -log_sigmoid(beta * (log_ratio_w - log_ratio_l))
```

### 记忆方法

四步拆解法：

1. **两个模型**：当前策略 $\pi_\theta$ 和参考策略 $\pi_{ref}$
2. **两个样本**：chosen（$y_w$）和 rejected（$y_l$）
3. **两两做差**：每个样本算 $\log\frac{\pi_\theta}{\pi_{ref}}$，这是"对数几率比"
4. **chosen 减 rejected**：鼓励 chosen 的几率比高于 rejected

口诀：**"四条 logprob，先减 ref 再减对，乘 beta 过 sigmoid，取负号"**

面试画图法：

```
π_θ(chosen)  ──┐
               ├── 差1 = log_θ_w - log_ref_w
π_ref(chosen) ─┘
                    差1 - 差2 → β × → sigmoid → -log
π_θ(rej)     ──┐
               ├── 差2 = log_θ_l - log_ref_l
π_ref(rej)   ─┘
```

### Python 实现

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def log_sigmoid(x):
    # 数值稳定: log(sigmoid(x)) = -log(1 + exp(-x))
    return -np.logaddexp(0, -x)

def dpo_loss(logp_chosen, logp_rejected,
             logp_ref_chosen, logp_ref_rejected,
             beta=0.1):
    """
    所有参数: scalar 或 [B]
    返回: scalar loss
    """
    log_ratio_w = logp_chosen - logp_ref_chosen
    log_ratio_l = logp_rejected - logp_ref_rejected
    loss = -log_sigmoid(beta * (log_ratio_w - log_ratio_l))
    return loss.mean()
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1):
    """
    所有参数: [B]
    """
    log_ratio_w = policy_chosen_logps - ref_chosen_logps
    log_ratio_l = policy_rejected_logps - ref_rejected_logps

    logits = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

---

## IPO（DPO 的替代损失）

### 一句话记忆

> **把 sigmoid 换成平方误差：$(\beta \cdot \Delta - 0.5)^2$。**

IPO 不用 log-sigmoid，而是直接回归到 0.5 的间隔。

### 伪代码

```
delta = log_ratio_chosen - log_ratio_rejected
loss = (delta - 1 / (2 * beta))^2
```

### PyTorch 实现

```python
def ipo_loss(log_ratio_w, log_ratio_l, beta=0.1):
    delta = log_ratio_w - log_ratio_l
    return ((delta - 1.0 / (2 * beta)) ** 2).mean()
```

---

## KTO（只需好/坏标签，不需要配对）

### 一句话记忆

> **好样本推高 log_ratio，坏样本压低 log_ratio，各自过 sigmoid。**

KTO 不需要 chosen-rejected 配对，只需要知道单条样本是好还是坏。

### 伪代码

```
log_ratio = log_pi(y|x) - log_pi_ref(y|x)

# 好样本：推高 log_ratio
loss_desirable = -log_sigmoid(beta * (log_ratio - z_ref))

# 坏样本：压低 log_ratio
loss_undesirable = -log_sigmoid(-beta * (log_ratio - z_ref))

loss = w_desirable * loss_desirable + w_undesirable * loss_undesirable
```

其中 `z_ref` 是 KL 估计的基线项。

### PyTorch 实现

```python
def kto_loss(log_ratio, is_desirable, z_ref=0.0, beta=0.1):
    """
    log_ratio: [B]  = log_pi(y|x) - log_ref(y|x)
    is_desirable: [B] bool  True = 好样本
    """
    loss = torch.zeros_like(log_ratio)

    desirable = is_desirable
    undesirable = ~is_desirable

    if desirable.any():
        loss[desirable] = -F.logsigmoid(
            beta * (log_ratio[desirable] - z_ref)
        )
    if undesirable.any():
        loss[undesirable] = -F.logsigmoid(
            -beta * (log_ratio[undesirable] - z_ref)
        )
    return loss.mean()
```

---

## SimPO（不需要 reference model）

### 一句话记忆

> **DPO 去掉 ref，换成 response length 归一化的 log 概率 + 隐式奖励 gamma。**

### 伪代码

```
logp_w = log_pi(chosen) / len(chosen)     # 长度归一化
logp_l = log_pi(rejected) / len(rejected)

loss = -log_sigmoid(beta * (logp_w - logp_l) - gamma)
```

### PyTorch 实现

```python
def simpo_loss(chosen_logps, rejected_logps,
               chosen_lengths, rejected_lengths,
               beta=2.0, gamma=0.5):
    logp_w = chosen_logps / chosen_lengths
    logp_l = rejected_logps / rejected_lengths
    logits = beta * (logp_w - logp_l) - gamma
    return -F.logsigmoid(logits).mean()
```

---

## DPO 家族对比速查

| 算法 | 需要 ref? | 需要配对? | 核心区别 |
|------|-----------|-----------|----------|
| DPO | 是 | 是 (chosen/rejected) | log-sigmoid，经典版 |
| IPO | 是 | 是 | 平方损失替代 log-sigmoid |
| KTO | 是 | 否 (好/坏标签) | 单样本级别优化 |
| SimPO | **否** | 是 | 长度归一化 + 隐式奖励偏移 |
| ORPO | **否** | 是 | odds ratio，合并 SFT + 对齐 |

---

## 易错点

| 易错 | 说明 |
|------|------|
| 四条 log_prob 搞混 | 记住：每个模型出两条（chosen + rejected），一共四条 |
| `log_sigmoid` 数值溢出 | PyTorch 的 `F.logsigmoid` 内置处理；手写时用 `logaddexp` |
| DPO 的 beta | beta 越大，对偏好差距越敏感，一般 0.1~0.5 |
| 忘了 detach ref | `ref_chosen_logps` 和 `ref_rejected_logps` 要 `.detach()`，不参与梯度 |
| chosen/rejected 反了 | 检查数据集：chosen 是人类偏好的那条 |
| IPO 没有 sigmoid | IPO 用平方损失，不需要 sigmoid，这是和 DPO 的关键区别 |
