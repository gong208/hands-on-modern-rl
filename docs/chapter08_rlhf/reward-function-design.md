# 8.4 Reward Model：教一个裁判

## 本节导读

**核心内容**

- 理解 Reward Model 为什么是 RLHF 的“裁判”，以及它和普通分类器有什么不同。
- 掌握 Bradley-Terry 偏好建模、pairwise loss、margin、accuracy、校准这些核心概念。
- 学会把 RM 分数放进混合奖励函数，并识别长度黑客、模板黑客和分布外高分。

**核心公式**

$$
P(y_w \succ y_l \mid x)
= \sigma(r_\theta(x,y_w)-r_\theta(x,y_l))
\quad \text{（Bradley-Terry：分数差越大，chosen 胜率越高）}
$$

$$
\mathcal{L}_{RM}
=-\mathbb{E}_{(x,y_w,y_l)}
\left[\log \sigma(r_\theta(x,y_w)-r_\theta(x,y_l))\right]
\quad \text{（RM pairwise loss）}
$$

$$
R_{total}(x,y)
= \hat r_{RM}(x,y)
+ \alpha R_{format}
+ \beta R_{correctness}
- \lambda R_{length}
- \eta R_{repeat}
\quad \text{（混合奖励：语义奖励 + 规则护栏）}
$$

> **先记住一句话**
>
> Reward Model 不是“真理机器”，它只是从偏好数据里学出来的裁判。PPO 会认真钻它的空子，所以 RM 训练和 RM 评估必须一起设计。

上一节我们准备了 SFT 数据和偏好数据。现在要训练 RLHF 的第二个关键 artifact：Reward Model。它接收 prompt 和 response，输出一个标量分数。本节会拆开这个黑盒，看看奖励函数到底怎么设计，为什么这么设计，以及设计不好会发生什么。

奖励函数是整个 RL 系统的"目标函数"——它定义了"什么是好"。策略网络不管你心里想的是什么，它只会朝奖励函数指的方向走。如果奖励函数说"字多就是好"，模型就会写长篇大论；如果奖励函数说"包含'我很乐意帮助您'就是好"，模型就会每句话都加上这句客套话。设计奖励函数就像给一个极度听话但毫无常识的助手下指令——你说什么它就做什么，包括你没想到的副作用。

## 8.4.1 奖励的谱系

奖励函数不是非此即彼的，而是一个从"纯规则"到"纯模型"的连续谱系：

```mermaid
flowchart LR
    R1["纯规则奖励\n（正则匹配）"] --> R2["混合奖励\n（规则 + 模型）"]
    R2 --> R3["纯模型奖励\n（RM 打分）"]

    R1 --- D1["✓ 确定性强\n✓ 零成本\n✗ 只能检查格式"]
    R3 --- D2["✓ 语义理解\n✓ 覆盖面广\n✗ 可被 hack"]

    style R1 fill:#e8f5e9,stroke:#2e7d32
    style R2 fill:#fff3e0,stroke:#e65100
    style R3 fill:#e3f2fd,stroke:#1565c0
```

**纯规则奖励**适合有客观标准答案的任务——数学题的最终答案对不对、代码能不能运行、输出格式是否合规。这类奖励完全确定，不可能被"hack"，但它只能检查表面形式，无法评估语义质量。

**纯模型奖励**就是训练一个奖励模型（RM），给它 $(prompt, response)$，它输出一个标量分数。RM 能理解语义——它知道"有帮助但语气生硬"和"礼貌但毫无内容"哪个更好。但 RM 有一个根本性的风险：它本身是一个模型，而模型可以被对抗性地利用。这就是后面会专门讨论的"奖励黑客"问题。

**混合奖励**是工业界最常用的方案——用 RM 覆盖语义层面，用规则覆盖 RM 捕捉不到的维度。典型的混合奖励函数长这样：

$$R_{total} = R_{RM} + \alpha \cdot R_{format} + \beta \cdot R_{length} + \gamma \cdot R_{correctness}$$

其中 $\alpha, \beta, \gamma$ 是需要调试的超参数。$R_{format}$ 检查格式规范度，$R_{length}$ 惩罚过长或过短的回答，$R_{correctness}$ 检查有客观答案的问题（数学、代码等）。

## 8.4.2 Bradley-Terry 模型

奖励模型的核心任务是从人类偏好对中学习一个评分函数。给定 prompt $x$ 和两个回答 $y_w$（chosen）和 $y_l$（rejected），人类标注员认为 $y_w$ 比 $y_l$ 好。RM 需要学出一个函数 $r_\theta(x, y)$，使得 $r_\theta(x, y_w) > r_\theta(x, y_l)$。

Bradley-Terry 模型给出了一个概率框架来描述这个过程：

$$P(y_w \succ y_l \mid x) = \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)$$

其中 $\sigma$ 是 sigmoid 函数 $\sigma(z) = \frac{1}{1 + e^{-z}}$。这个公式的直觉很清楚：如果 RM 给 $y_w$ 的分数比 $y_l$ 高很多，那么 $y_w$ 被选中的概率就接近 1；如果两个分数差不多，概率就接近 0.5。

对应的训练损失函数是负对数似然：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right) \right]$$

让我们逐项认识这个公式：

| 符号                 | 角色          | 大白话                         |
| -------------------- | ------------- | ------------------------------ |
| $r_\theta(x, y)$     | 奖励函数      | 给 (问题, 回答) 打一个分数     |
| $y_w$                | chosen 回答   | 标注员认为更好的回答           |
| $y_l$                | rejected 回答 | 标注员认为较差的回答           |
| $\sigma(\cdot)$      | sigmoid       | 把分数差映射到 $(0, 1)$ 的概率 |
| $\log \sigma(\cdot)$ | 对数似然      | 优化目标——让 RM 更大概率选对   |

```python
# ==========================================
# 奖励模型训练：Bradley-Terry 损失
# ==========================================
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """奖励模型：输入 (prompt, response)，输出标量分数"""
    def __init__(self, base_model, hidden_dim=1024):
        super().__init__()
        # 基座模型（通常用 SFT 模型或较小的预训练模型）
        self.base = base_model
        # 在最后一层隐藏状态上加一个线性头，输出标量
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        # 取最后一个 token 的隐藏状态
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # (batch, hidden_dim)
        reward = self.reward_head(last_hidden)  # (batch, 1)
        return reward.squeeze(-1)


def bradley_terry_loss(rm, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    """Bradley-Terry 偏好损失"""
    r_chosen = rm(chosen_ids, chosen_mask)     # chosen 回答的分数
    r_rejected = rm(rejected_ids, rejected_mask)  # rejected 回答的分数

    # 核心：让 chosen 的分数比 rejected 高
    loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
    return loss
```

这段代码里有几个值得注意的工程细节。首先，RM 的基座模型通常选择与策略模型相同架构但较小规模的模型——比如 7B 策略用 3B 做 RM。这不是因为大 RM 不好，而是因为 RM 在 RL 训练阶段需要频繁推理（每一步都要打分），太大的 RM 会严重拖慢训练速度。其次，RM 通常在 SFT 模型基础上微调，而不是从 base 模型开始——SFT 模型已经理解了"回答"的格式，微调只需要学会"哪个回答更好"。

### 手算一次 Bradley-Terry loss

公式不难，但最好亲手算一次。假设同一个 prompt 下，RM 给 chosen 和 rejected 的分数是：

$$
r_\theta(x,y_w)=2.0,\qquad r_\theta(x,y_l)=0.5
$$

分数差是：

$$
\Delta r = 2.0 - 0.5 = 1.5
$$

chosen 胜出的概率是：

$$
\sigma(1.5)=\frac{1}{1+e^{-1.5}}\approx 0.818
$$

loss 是：

$$
-\log 0.818 \approx 0.201
$$

这说明 RM 对这对样本比较有信心，而且方向正确。如果它给反了，比如 chosen 分数 0.5，rejected 分数 2.0：

$$
\Delta r=-1.5,\qquad \sigma(-1.5)\approx 0.182,\qquad -\log 0.182\approx 1.704
$$

loss 会大很多，梯度会推动 chosen 分数上升、rejected 分数下降。

### 为什么不用绝对打分

你可能会问：为什么不让标注员直接给回答打 1 到 10 分？原因是人类对绝对分数很不稳定。同一个回答，有人给 7 分，有人给 9 分；同一个人早上和晚上给分也可能不同。但人类更擅长做相对判断：A 是否比 B 更好。

偏好建模利用的正是这个特点：

| 标注方式   | 人类是否容易一致 | 训练信号         | 主要问题               |
| ---------- | ---------------- | ---------------- | ---------------------- |
| 绝对打分   | 较难             | 回归分数         | 标注员尺度不一致       |
| 二选一偏好 | 较容易           | pairwise ranking | 只能比较同 prompt 回答 |
| 多回答排序 | 中等             | 可拆成多个偏好对 | 标注成本更高           |

实际标注里常见做法是：同一个 prompt 采样 4 到 9 个回答，让标注员排序，再把排序结果拆成多个 chosen/rejected 对。

## 8.4.3 奖励粒度

RM 给一个回答打一个分数，这个分数应该怎么分配？是给整个回答一个总分，还是给每个 token 单独打分，还是按推理步骤分段打分？这就是奖励粒度的问题。

| 粒度           | 方式                | 优势               | 劣势                       | 代表方法        |
| -------------- | ------------------- | ------------------ | -------------------------- | --------------- |
| Sequence-level | 整个回答一个分数    | 简单，稳定         | 无法区分好回答中哪些部分好 | PPO, GRPO       |
| Step-level     | 按推理步骤分段      | 折中精细度与可行性 | 需要步骤分割器             | PRM（过程监督） |
| Token-level    | 每个 token 独立分数 | 最精细             | 训练成本高，信号噪声大     | RLHF 早期尝试   |

实践中最常用的是 **sequence-level** 加上 **规则辅助**。PPO 和 GRPO 默认都是给整个回答一个总奖励分数，然后通过规则奖励来补充 token-level 的信号（比如格式奖励检查每个 token 是否符合特定格式）。

Step-level 奖励是一个值得关注的方向，常被称为 Process Reward Model（PRM）或过程监督。OpenAI 在 2023 年的过程监督研究中比较了 outcome supervision 和 process supervision，并发布了 PRM800K step-level feedback 数据集。[^process_supervision] 它的直觉很好理解：如果最终答案错了，但中间某几步推理是正确的，step-level 奖励可以强化这些正确步骤；sequence-level 奖励会把整条推理链一起惩罚。

```python
# ==========================================
# 不同粒度的奖励计算
# ==========================================
def sequence_reward(rm, prompt, response):
    """Sequence-level：整个回答一个分数"""
    return rm.score(prompt, response)  # 一个标量

def step_reward(rm, prompt, reasoning_steps):
    """Step-level：每个推理步骤一个分数"""
    step_rewards = []
    for i, step in enumerate(reasoning_steps):
        # 对前 i+1 步的累积内容打分
        partial = "\n".join(reasoning_steps[:i+1])
        step_rewards.append(rm.score(prompt, partial))
    return step_rewards  # 一个列表

def combined_reward(rm, prompt, response, reasoning_steps):
    """混合：sequence-level RM + step-level 规则"""
    r_rm = sequence_reward(rm, prompt, response)
    r_format = 0.2 if validate_format(response) else 0.0  # 格式奖励
    r_correct = 1.0 if check_answer_correct(prompt, response) else 0.0  # 正确性奖励
    r_length = -0.01 * max(0, len(response) - 500)  # 长度惩罚

    return r_rm + r_format + r_correct + r_length
```

<details>
<summary>思考题：为什么 token-level 奖励在实际中很少使用？</summary>

Token-level 奖励要求对每一个 token 都给出一个独立的分数。理论上这最精细——它可以告诉模型"第 3 个 token 选得好，第 7 个 token 选得不好"。但实际中有两个根本性的困难：

第一，**标注成本**。人类标注员可以对整个回答做排序（"A 比 B 好"），但无法对每个 token 做精细标注（"这个 token 好还是不好"）。这意味着 token-level 奖励几乎必须通过模型来生成，而模型生成的 token-level 信号本身就有噪声。

第二，**信用分配问题**。一个回答好或不好，通常是多个 token 协同作用的结果。单独看每个 token 很难判断它的贡献——就像你不能通过单独品尝每种调料来判断一道菜好不好吃。Step-level 奖励是一种折中方案：它比 sequence-level 更精细，又避免了 token-level 的信用分配难题。

</details>

### LLM 里的信用分配问题

Sequence-level RM 有一个根本难题：它只在回答结束后给分。假设模型生成了 200 个 token，RM 给了低分。是哪一段导致低分？

| 可能原因           | 发生位置    | sequence reward 能否直接指出 |
| ------------------ | ----------- | ---------------------------- |
| 开头误解了用户意图 | 前 20 token | 不能                         |
| 中间推理跳步       | 中间段落    | 不能                         |
| 最后答案写错       | 结尾        | 不能                         |
| 语气过度自信       | 全文        | 不能                         |

这就是 PPO-RLHF 里 Critic 和 advantage 估计存在的原因之一：它们不能完美解决信用分配，但能把“整段回答好不好”的信号更平滑地传回 token 级别更新。后面的 GRPO、RLVR、过程奖励，本质上也都在不同方向上处理这个问题。

## 8.4.4 规则奖励与模型奖励

选择规则奖励还是模型奖励，取决于你的任务类型和约束条件。

|              | 规则奖励                             | 模型奖励（RM）                 |
| ------------ | ------------------------------------ | ------------------------------ |
| **成本**     | 几乎为零                             | 训练 + 推理成本                |
| **可靠性**   | 确定性强，不可被 hack                | 可被对抗性利用                 |
| **语义理解** | 无，只能检查格式                     | 有，能理解内容质量             |
| **泛化能力** | 差，每换一个任务要写新规则           | 好，同一个 RM 可以评估多种回答 |
| **适用场景** | 数学/代码/格式检查等有客观标准的任务 | 对话/创意/安全等主观偏好任务   |
| **典型用法** | 作为混合奖励的"底线"                 | 作为混合奖励的"主体"           |

在下一章的 RLVR（Reinforcement Learning with Verifiable Rewards）场景中，规则奖励会成为主力。数学题的答案对不对可以用代码验证，代码能不能运行可以直接执行。但在开放域对话、创意写作、安全对齐等场景中，没有客观标准，模型奖励不可替代。

一个实用的经验法则是：**能用规则奖励的地方就用规则奖励，规则覆盖不到的地方用模型奖励补**。这样做的好处是规则奖励提供了一个"安全网"——即使 RM 被hack了，规则奖励仍然能确保基本的格式和正确性。

## 8.4.5 RM 训练的工程细节

训练一个靠谱的 RM，除了 Bradley-Terry 损失之外，还有很多工程细节需要注意。

**数据标注**方面，标注员通常被要求对同一个 prompt 的 4-9 个回答进行排序，而不是打绝对分。人类更擅长比较——"A 比 B 好"比"A 值 8 分"更容易达成一致。排序数据可以拆成多对偏好对来训练 RM。

**训练稳定性**方面，RM 训练有几个关键的超参数：

```python
# ==========================================
# RM 训练的关键配置
# ==========================================
rm_config = {
    # 基座模型：通常用 SFT 后的较小模型
    "base_model": "sft_model_3b",

    # 学习率：比 SFT 更保守
    "learning_rate": 5e-6,  # SFT 通常用 1e-5 到 2e-5

    # 学习率调度：线性 warmup + 余弦衰减
    "warmup_steps": 100,
    "lr_scheduler": "cosine",

    # 梯度裁剪：防止梯度爆炸
    "max_grad_norm": 1.0,

    # 批大小：偏好对的数量
    "batch_size": 128,  # 每个批次 128 对 (chosen, rejected)

    # 训练轮数：通常只需 1-2 个 epoch
    "epochs": 1,  # RM 容易过拟合，不要训太多轮
}
```

RM 特别容易过拟合——因为偏好数据通常只有几万到几十万对，而 RM 的参数量可能有几十亿。1 个 epoch 通常是最佳选择，超过 2 个 epoch 往往会导致验证集上的准确率开始下降。

**RM 的评估**也是一门学问。最直接的指标是"在 held-out 偏好对上的准确率"——RM 能在多大程度上复现人类标注的偏好。但这个指标有一个盲区：它衡量的是"排序对不对"，不是"打分准不准"。一个 RM 可能在所有排序对上都选对了，但给 chosen 的分数和 rejected 的分数差距很小——这意味着它的信号太弱，在 RL 阶段起不到足够的引导作用。因此，实践中还需要关注 RM 分数的区分度（margin）。

### 数据切分：不能按 pair 随机切

RM 训练里一个隐蔽坑是数据切分方式。如果同一个 prompt 下有 6 个候选回答，你把拆出来的 pair 随机分到 train 和 eval，就会发生泄露：训练集和验证集共享同一个 prompt，甚至共享部分回答。这样得到的 eval accuracy 会偏乐观。

更稳妥的做法是按 prompt 切分：

```python
def split_by_prompt(items, eval_ratio=0.1):
    """
    items: [{"prompt_id": str, "prompt": str, "chosen": str, "rejected": str}, ...]
    """
    import random
    prompt_ids = sorted({item["prompt_id"] for item in items})
    random.shuffle(prompt_ids)

    n_eval = int(len(prompt_ids) * eval_ratio)
    eval_ids = set(prompt_ids[:n_eval])

    train, eval_ = [], []
    for item in items:
        if item["prompt_id"] in eval_ids:
            eval_.append(item)
        else:
            train.append(item)
    return train, eval_
```

按 prompt 切分能更真实地回答：RM 遇到没见过的新问题时，偏好排序是否还能泛化？

### RM 分数尺度：PPO 前必须校准

RM 训练只关心分数差，不关心绝对尺度。也就是说，如果一个 RM 输出 $(2, 1)$，另一个输出 $(20, 10)$，它们在排序上都对，但 PPO 阶段感受到的奖励尺度完全不同。

这会直接影响训练稳定性：

| RM 分数尺度 | PPO 中可能发生什么                       |
| ----------- | ---------------------------------------- |
| 太小        | reward 信号被 KL 惩罚淹没，Actor 学不动  |
| 太大        | reward 压过 KL，Actor 快速偏离 reference |
| 漂移严重    | 不同 batch 的优势估计不稳定              |

常见做法是在固定校准集上做标准化：

```python
class RewardNormalizer:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, reward):
        return (reward - self.mean) / (self.std + self.eps)
```

这里的 mean/std 应该来自固定校准集，而不是训练过程中随便用当前 batch 更新。否则 reward 尺度会随着 Actor 分布一起漂，排查问题会很痛苦。

### RM 评估看什么

一个比较完整的 RM 报告至少包含：

| 指标                      | 含义                                  | 典型用途                 |
| ------------------------- | ------------------------------------- | ------------------------ |
| Pairwise accuracy         | held-out 偏好对上 chosen 分数是否更高 | 检查排序能力             |
| Mean margin               | chosen 和 rejected 平均分数差         | 检查信号强度             |
| Margin distribution       | 分差分布是否健康                      | 找出“勉强排对”的样本     |
| Reward-length correlation | 分数是否过度依赖长度                  | 检查长度黑客风险         |
| Domain breakdown          | 各任务域 accuracy/margin              | 找出偏科                 |
| Calibration samples       | 人工看高分/低分样本                   | 检查 RM 是否符合人类直觉 |

一个轻量计算函数：

```python
def rm_eval_metrics(r_chosen, r_rejected, chosen_lengths, rejected_lengths):
    import numpy as np

    margin = np.asarray(r_chosen) - np.asarray(r_rejected)
    accuracy = float((margin > 0).mean())

    rewards = np.concatenate([r_chosen, r_rejected])
    lengths = np.concatenate([chosen_lengths, rejected_lengths])
    length_corr = float(np.corrcoef(rewards, lengths)[0, 1])

    return {
        "pairwise_accuracy": accuracy,
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "length_reward_corr": length_corr,
    }
```

如果 `length_reward_corr` 很高，就要回头检查偏好数据：是不是 chosen 普遍比 rejected 更长？如果是，RM 可能学到“长就是好”，而不是“有帮助就是好”。

## 8.4.6 奖励黑客

奖励函数设计里的最大风险不是 RM accuracy 低，而是 RM 有系统性盲区。PPO 阶段的 Actor 不是随机用户，它会主动搜索让 RM 给高分的输出分布。如果 RM 偏爱某种表面模式，Actor 会把这种模式推到极端。

| RM 盲区       | Actor 可能学到 | 表面指标           | 真实问题     |
| ------------- | -------------- | ------------------ | ------------ |
| 偏爱长回答    | 越写越长       | reward 上升        | 信息密度下降 |
| 偏爱礼貌模板  | 反复客套       | judge 可能短期喜欢 | 内容变空     |
| 偏爱 Markdown | 到处堆标题列表 | 格式更整齐         | 不一定更准确 |
| 安全数据过窄  | 机械拒答       | 风险下降           | 可用性下降   |
| 专业术语高分  | 术语堆砌       | 看起来专业         | 幻觉更多     |

所以 RM 训练完不是“丢给 PPO 就完了”。你要在 PPO 前先做对抗性测试：

```python
stress_cases = [
    ("空回答", ""),
    ("超长废话", "这个问题非常重要。" * 200),
    ("固定模板", "当然可以。以下是一些建议：\n" * 50),
    ("事实错误但自信", "PPO 是 1980 年提出的确定性搜索算法。"),
    ("正确但简短", "PPO 用裁剪限制新旧策略差异，避免更新过猛。"),
]

for name, response in stress_cases:
    print(name, reward_model.score(prompt, response))
```

如果“超长废话”比“正确但简短”分数高，先别跑 PPO。PPO 只会把这个问题放大。

## 8.4.7 奖励函数设计的工程检查清单

把这一节的内容整合成一个实用的检查清单，方便你在设计自己的奖励函数时逐项对照：

| 检查项    | 问题                                                   | 通过标准                         |
| --------- | ------------------------------------------------------ | -------------------------------- |
| 奖励粒度  | 你选择的是什么粒度的奖励？                             | 有明确的理由说明为什么选这个粒度 |
| 混合奖励  | 是否同时使用规则奖励和模型奖励？                       | 至少包含一个规则奖励作为底线     |
| 长度惩罚  | 是否有防止模型写太长的机制？                           | 有明确的长度惩罚项               |
| 重复惩罚  | 是否有防止模型重复废话的机制？                         | 有 n-gram 重复率检测             |
| RM 区分度 | RM 的 chosen/rejected 分数差距是否足够大？             | 平均 margin > 1.0                |
| RM 过拟合 | RM 是否在验证集上表现良好？                            | 验证集准确率 > 65%               |
| 边界情况  | 奖励函数对极端输入（空回答、超长回答）的行为是否合理？ | 边界情况有明确处理               |
| 分数校准  | RM 输出尺度是否适合 PPO？                              | 固定校准集均值、方差稳定         |
| 领域分解  | 不同任务域是否表现一致？                               | 没有明显偏科或安全退化           |

这个检查清单不能保证你的奖励函数完美无缺，但能帮你避开最常见的坑。记住：奖励函数是 RL 系统的北极星——它指的方向不对，模型跑得越快偏得越远。

奖励函数设计好了，下一步就是把 SFT 模型、Reward Model、Reference 和 Critic 接成 PPO-RLHF 训练循环。让我们进入下一节——[PPO-RLHF：按奖励练习](./ppo-rlhf-loop)。

## 本节小结

Reward Model 把人类偏好压缩成一个标量分数。这个压缩非常有用，因为 PPO 需要一个可优化的奖励；但它也很危险，因为开放式回答的质量无法被一个完美标量完全表达。

训练 RM 时，Bradley-Terry loss 只解决了“偏好对怎么变成梯度”的问题。真正让 RM 可用的，是数据切分、margin、分数校准、长度相关性检查、对抗样本和人工抽检。一个好的 RM 不只是 held-out accuracy 高，还要在 PPO 会探索到的区域里尽量不胡乱给高分。

## 练习

1. 手算两组 RM 分数的 Bradley-Terry loss：$(r_w,r_l)=(1.0,0.0)$ 和 $(0.0,1.0)$。
2. 写 5 条 stress case，测试一个 RM 是否偏爱长回答或固定模板。
3. 解释为什么 RM 训练要按 prompt 切分 train/eval，而不是按 pair 随机切分。

## 参考文献

[^process_supervision]: OpenAI. [Improving mathematical reasoning with process supervision](https://openai.com/research/improving-mathematical-reasoning-with-process-supervision), 2023.
