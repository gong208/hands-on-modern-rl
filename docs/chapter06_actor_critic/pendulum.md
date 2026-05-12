# 6.5 动手：Pendulum 摆杆平衡

> **本节目标**：用 Actor-Critic 训练 `Pendulum-v1`，理解策略网络如何输出连续动作（高斯分布），体验 AC 在连续控制中的优势。

> **本节代码**：[actor_critic_pendulum.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter06_actor_critic/actor_critic_pendulum.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter06_actor_critic/requirements.txt)

前几章的实验都是离散动作——CartPole 的"左/右"、LunarLander 的四个喷口。但 Actor-Critic 的真正优势在于**连续动作空间**——这正是 DQN 做不了的事。

`Pendulum-v1` 是最简单的连续控制任务：一根杆子挂在轴上，智能体可以施加连续的力矩 $[-2, 2]$ 来让杆子摆到正上方并保持平衡。状态只有 3 维（$\cos\theta$, $\sin\theta$, 角速度），动作只有 1 维（力矩），但它是连续的——不可能给每个力矩值都算一个 $Q$ 值然后取 $\arg\max$。

## 连续动作的策略网络

离散动作的策略网络输出 Softmax 概率。连续动作的策略网络输出**高斯分布**的参数——均值 $\mu$ 和标准差 $\sigma$：

```python
class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor 头：输出高斯分布的均值
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        # Actor 头：输出 log std（用 log 保证 std > 0）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic 头：输出 V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        mu = torch.tanh(self.mu_head(features)) * 2  # 缩放到 [-2, 2]
        std = torch.exp(self.log_std)
        value = self.critic_head(features)
        return mu, std, value
```

关键区别：Actor 不再输出概率分布，而是输出高斯分布的 $\mu$ 和 $\sigma$。动作从 $\mathcal{N}(\mu, \sigma^2)$ 中采样，然后用 `tanh` 压缩到 $[-2, 2]$。

## 运行训练

```bash
pip install -r code/chapter06_actor_critic/requirements.txt
python code/chapter06_actor_critic/actor_critic_pendulum.py
```

训练通常在 200-300 个 episode 内收敛。成功的标志是杆子能稳定在正上方附近——奖励从 -1000 左右（随机策略）上升到接近 0（最优策略，Pendulum 的奖励上限是 0）。

## 为什么 DQN 做不了 Pendulum

DQN 需要用 $\arg\max_a Q(s,a)$ 选动作。Pendulum 的动作是连续值 $[-2, 2]$——有无穷多个候选，不可能逐一比较。DQN 的做法是把 $[-2, 2]$ 离散化成比如 20 个格子，但这样精度损失严重。

Actor-Critic 直接从 $\mathcal{N}(\mu, \sigma^2)$ 采样，天然适用于连续空间。这就是本章开头说的："Actor 直接输出动作概率，天然适用于连续动作空间"的具体体现。

下一节，我们挑战一个更复杂的连续控制任务：[动手：BipedalWalker 双足行走](./bipedalwalker)。
