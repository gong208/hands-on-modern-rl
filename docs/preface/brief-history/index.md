# 强化学习简史

如果我们在 2010 年代初询问一位 AI 研究员“什么是强化学习”，他大概会给你画一个智能体与环境交互的反馈循环图，并告诉你这主要用于机器人控制和下棋。但如果我们将时间的指针拨回一个世纪前，或者快进到今天的大模型时代，你会发现强化学习（Reinforcement Learning, RL）经历了一场波澜壮阔的演变——它从心理学家的动物实验出发，一步步成长为驱动当今最先进 AI 系统的核心引擎。

在开始我们的代码实践之前，不妨先花几分钟，快速回顾一下这段跨越百年的简史。了解这些里程碑，能帮助你更好地理解为什么现代 RL 算法会设计成今天的样子。

## 1. 启蒙与奠基：从心理学到数学框架（1890s - 1950s）

强化学习的思想最早并非诞生于计算机科学，而是来自**心理学和神经科学**。
1898 年，心理学家爱德华·桑代克（Edward Thorndike）通过著名的"猫的迷笼实验"提出了**效果律（Law of Effect）**：如果一个行为带来了好的结果，这个行为就会被强化；反之则被弱化。这正是"试错学习（Trial-and-Error）"的本源。

![Thorndike's Puzzle Box](./images/puzzle_box.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 1：桑代克设计的迷笼（Puzzle Box）。来源：<a href="https://commons.wikimedia.org/wiki/File:Original_%22Puzzle_Box%22_Apparatus_Design.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

半个多世纪后，随着控制论的兴起，这种生物本能开始被严谨地数学化。1957 年，理查德·贝尔曼（Richard Bellman）提出了**马尔可夫决策过程（MDP）**与**贝尔曼方程（Bellman Equation）** [^1]。他第一次为"序列决策问题"建立了一套完美的数学框架，将试错学习转化为了"最大化长期累积奖励"的最优化问题。这标志着强化学习正式拥有了坚实的理论根基。

## 2. 理论成型：时序差分与无模型学习（1980s - 1990s）

贝尔曼的动态规划虽然在数学上无懈可击，但它要求完全已知环境的模型，且面临着严重的"维度灾难"。为了让智能体在未知环境中学习，先驱者们开始寻找新的出路。

- **1988 年**，被誉为"强化学习之父"的理查德·萨顿（Richard Sutton）系统性地提出了**时序差分学习（Temporal Difference, TD）** [^2]。它巧妙地结合了蒙特卡洛采样和动态规划的自举特性，让智能体可以在没有完整环境模型的情况下边走边学。
- **1989 年**，克里斯·沃特金斯（Chris Watkins）在他的博士论文中提出了著名的 **Q-Learning** 算法 [^3]。这是一种无模型（Model-Free）的离策略算法，至今仍是 RL 入门的第一课。
- **1992 年**，IBM 的杰拉尔德·特萨罗（Gerald Tesauro）开发了 **TD-Gammon** [^4]。通过将 TD 算法与一个浅层神经网络结合，它在西洋双陆棋中达到了人类世界冠军的水平。这是神经网络与 RL 结合的早期成功典范。

![TD-Gammon / Backgammon](./images/backgammon.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 2：西洋双陆棋（Backgammon），TD-Gammon 攻克的经典游戏。来源：<a href="https://commons.wikimedia.org/wiki/File:Backgammon_lg.jpg" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

1998 年，Sutton 和 Barto 出版了影响深远的经典教材《强化学习：一个介绍》（_Reinforcement Learning: An Introduction_） [^5]，现代强化学习的学科框架正式成型。

## 3. 深度革命：当 RL 遇见深度学习（2013 - 2019）

进入 21 世纪后，尽管 RL 理论日益完善，但传统的表格型方法和线性函数近似根本无法处理真实世界中高维、复杂的输入（如图像）。直到深度学习的爆发，RL 才真正迎来了它的"高光时刻"。

- **2013 年**，DeepMind 提出了**深度 Q 网络（DQN）** [^6]，首次将深度神经网络与 RL 完美结合，让 AI 仅凭屏幕像素就能学会在多款 Atari 街机游戏中超越人类。深度强化学习（Deep RL）的时代正式拉开帷幕。

![DQN Atari Performance](./images/dqn_atari.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 3：DQN 在数十款 Atari 游戏中的表现，大部分超越了人类专业玩家。来源：<a href="https://research.google/blog/from-pixels-to-actions-human-level-control-through-deep-reinforcement-learning/" target="_blank" rel="noopener noreferrer">Google Research Blog</a></em>
</div>

- **2016 年**，注定载入史册的一年。DeepMind 的 **AlphaGo** [^7] 结合了深度强化学习与蒙特卡洛树搜索，以 4:1 击败了围棋世界冠军李世石。这一事件不仅震惊了世界，也让 RL 第一次以极其震撼的方式进入了公众视野。

![AlphaGo](./images/alphago.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 4：AlphaGo 击败欧洲围棋冠军樊麾的对局截图。来源：<a href="https://commons.wikimedia.org/wiki/File:AlphaGo_Fan_Huiren_aurka.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

- **2017 年**，OpenAI 提出了 **PPO（近端策略优化，Proximal Policy Optimization）** 算法 [^8]。相比于早期策略梯度方法的高方差和脆弱性，PPO 在训练稳定性和采样效率之间找到了绝佳的平衡。由于其易于调参和出色的鲁棒性，PPO 迅速成为了工业界的默认标准算法。随后 OpenAI 使用基于 PPO 的大规模分布式系统 **OpenAI Five** 击败了 DOTA 2 的世界冠军团队。

## 4. 大模型时代：对齐与推理的新范式（2020s 至今）

就在人们以为 RL 的应用边界主要局限于游戏和机器人控制时，大语言模型（LLM）的崛起为 RL 赋予了全新的使命——**对齐（Alignment）**与**推理（Reasoning）**。

- **2022 年**，OpenAI 发布了 ChatGPT。其背后的核心功臣正是 **RLHF（基于人类反馈的强化学习）** [^9]。通过训练一个奖励模型来模拟人类偏好，再用 PPO 算法优化语言模型，RL 成功地让 LLM 从"能接话的统计机器"变成了"懂分寸的智能助手"。

![ChatGPT](./images/chatgpt.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 5：ChatGPT 使得 RLHF 技术闻名于世。来源：<a href="https://commons.wikimedia.org/wiki/File:ChatGPT.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons</a></em>
</div>

- **2023 年**，斯坦福大学等提出了 **DPO（直接偏好优化）** [^10]。研究者们发现，可以绕过繁琐的奖励模型训练，直接用一个简单的分类损失函数在人类偏好数据上微调语言模型。DPO 极大地降低了 RLHF 的工程门槛，迅速席卷了开源社区。
- **2024 - 2025 年**，随着 OpenAI o1 和 DeepSeek-R1 [^11] 等推理模型的惊艳亮相，强化学习再次进化。特别是 **DeepSeek-R1-Zero 证明了在有明确客观规则（如数学对错、代码编译）的场景下，完全可以抛弃传统的 SFT（监督微调）冷启动，直接让 Base 模型进行纯粹的强化学习（Pure RL）。** 这一过程不仅打破了“必须先 SFT 才能做 RL”的刻板印象，更让模型自主涌现出了长思维链（CoT）和顿悟（a-ha moment）能力。DeepSeek 采用的 **GRPO（群体相对策略优化）** 算法，去除了传统 PPO 中极其消耗显存的 Critic 网络，直接通过组内相对奖励来优化策略。这种轻量级架构使得在大规模集群上进行纯 RL 强化推理能力成为现实。

## 小结

从桑代克的迷笼，到贝尔曼的方程；从雅达利游戏机里的 DQN，到今天云端集群里飞速迭代的 DPO 和 GRPO。强化学习的历史，就是一部智能体**"从环境中学习、从反馈中进化、从单机走向超级模型"**的史诗。

今天，强化学习已经不再是象牙塔里的理论玩具，它是通向通用人工智能（AGI）的必经之路。在接下来的章节中，我们将沿着这段历史的脉络，从第一行代码开始，亲手将这些伟大的算法实现出来。

## 参考文献

[^1]: Bellman, R. (1957). A Markovian Decision Process. _Journal of Mathematics and Mechanics_, 6(5), 679-684. [DOI](https://doi.org/10.1512/iumj.1957.6.56038)

[^2]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44. [PDF](http://incompleteideas.net/papers/sutton-88.pdf)

[^3]: Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. _PhD Thesis, King's College, Cambridge_. [PDF](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)

[^4]: Tesauro, G. (1995). Temporal difference learning and TD-Gammon. _Communications of the ACM_, 38(3), 58-68. [DOI](https://doi.org/10.1145/203330.203343)

[^5]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. [在线阅读](http://incompleteideas.net/book/the-book.html)

[^6]: Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. _arXiv preprint_. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

[^7]: Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. _Nature_, 529(7587), 484-489. [DOI](https://doi.org/10.1038/nature16961)

[^8]: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. _arXiv preprint_. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[^9]: Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. _arXiv preprint_. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

[^10]: Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv preprint_. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

[^11]: DeepSeek-AI, et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. _arXiv preprint_. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
