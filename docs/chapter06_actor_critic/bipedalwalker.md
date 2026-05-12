# 6.6 动手：BipedalWalker 双足行走

> **本节目标**：用 Actor-Critic 训练 `BipedalWalker-v3`，观察策略如何学会协调 4 个连续关节实现双足行走——这是 Actor-Critic 处理高维连续控制的标志性任务。

> **本节代码**：[actor_critic_bipedalwalker.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter06_actor_critic/actor_critic_bipedalwalker.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter06_actor_critic/requirements.txt)

上一节的 Pendulum 只有 1 维连续动作、3 维状态。BipedalWalker 把复杂度提升了一个量级：24 维状态（关节角度、角速度、地面接触传感器等），4 维连续动作（髋关节和膝关节各两个），目标是让一个双足机器人学会走路。

## 环境：BipedalWalker-v3

```
        O          ← 头部
       /|\
      / | \        ← 躯干
     /  |  \
    🔶   🔶       ← 髋关节
    |     |        ← 大腿
    🔷   🔷       ← 膝关节
    |     |        ← 小腿
   ___   ___       ← 脚
```

| 属性        | 值                                                  |
| ----------- | --------------------------------------------------- |
| 状态维度    | 24（躯干角度、角速度、关节状态、10 个激光雷达测距） |
| 动作维度    | 4（左髋、左膝、右髋、右膝的力矩，连续值 $[-1, 1]$） |
| 奖励        | 前进距离 + 存活惩罚 - 能量消耗                      |
| 终止        | 摔倒（头部触地）或到达终点                          |
| "Work" 标志 | 平均奖励 > 300                                      |

## 运行训练

```bash
pip install -r code/chapter06_actor_critic/requirements.txt
python code/chapter06_actor_critic/actor_critic_bipedalwalker.py
```

BipedalWalker 比 Pendulum 和 CartPole 难得多，训练可能需要 1000-3000 个 episode。成功的标志是机器人能稳定行走不摔倒，平均奖励超过 300。

## 从 Pendulum 到 BipedalWalker

|             | Pendulum   | BipedalWalker         |
| ----------- | ---------- | --------------------- |
| 状态维度    | 3          | 24                    |
| 动作维度    | 1          | 4                     |
| 训练时间    | 几分钟     | 几十分钟到几小时      |
| "Work" 标准 | 奖励接近 0 | 奖励 > 300            |
| 难点        | 单关节控制 | 多关节协调 + 动态平衡 |

BipedalWalker 的核心挑战是**协调**：4 个关节需要同时以正确的方式发力，任何关节的动作不协调都会导致摔倒。这正是 Actor-Critic 擅长的——Actor 可以同时输出 4 个关节的动作，Critic 评估整体状态的好坏，两者协作让策略逐步学会协调运动。

## 本章小结

本章从 REINFORCE 的高方差问题出发，引入了 Actor-Critic 架构：用 Critic 网络估计 $V(s)$ 提供低方差的优势信号，用 Actor 网络做决策。从 CartPole（离散）到 Pendulum（1 维连续）再到 BipedalWalker（4 维连续），我们看到 Actor-Critic 架构随任务复杂度增长而持续有效。

下一章，我们将解决 Actor-Critic 的另一个问题——训练不稳定——引出 PPO 算法：[第 7 章 PPO](../chapter07_ppo/intro)。
