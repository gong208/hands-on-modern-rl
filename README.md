<div align="center">
  <h1>Hands-On Modern RL</h1>
  <p><strong>现代强化学习实战</strong></p>
  <p><em>From Code to Theory — 从代码到原理</em></p>

  <p>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/stargazers"><img src="https://img.shields.io/badge/Stars-0-eab676?style=for-the-badge&logo=github" alt="Stars" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/network/members"><img src="https://img.shields.io/badge/Forks-0-87a96b?style=for-the-badge&logo=github" alt="Forks" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/issues"><img src="https://img.shields.io/badge/Issues-0-c780e8?style=for-the-badge&logo=github" alt="Issues" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/pulls"><img src="https://img.shields.io/badge/Pull%20Requests-0-7ea9e1?style=for-the-badge&logo=github" alt="Pull Requests" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-e26d5c?style=for-the-badge" alt="CC BY-NC-SA 4.0 License" /></a>
  </p>

  <p>
    <a href="#课程概述">课程概述</a> &middot;
    <a href="#课程大纲">课程大纲</a> &middot;
    <a href="#本地运行环境">本地运行环境</a> &middot;
    <a href="#参与贡献">参与贡献</a>
  </p>
</div>

---

> 作者水平有限，加上使用了 AI 辅助写作，内容可能存在错误或不准确的地方，欢迎大家指出。希望这门课程能让强化学习不再那么令人望而生畏，让更多普通人也有攀登智能上限的勇气。

## 课程概述

强化学习（Reinforcement Learning）是机器学习中一个核心但门槛较高的分支。传统的教学路径通常从马尔可夫决策过程（MDP）和贝尔曼方程的形式化定义出发，这对许多学习者的耐心和理解力构成了不必要的挑战。

本课程采用另一种路径：**先建立直觉，再引入形式化**。我们认为，当学习者亲手实现一个算法、观察到它的行为，再回过头理解其背后的数学结构时，学习过程会更加自然，理解也会更加持久。

具体而言，课程围绕以下四条线索展开：

1. **代码先行**——第一章即动手训练 CartPole 智能体，第二章用 DPO 微调大语言模型，通过实验建立对状态、动作、奖励、策略的直觉。
2. **算法拆解**——逐行精读 DQN、PPO、DPO、GRPO 等工业级算法的实现，追问每一步设计选择背后的动机，而非仅仅调用 API。
3. **理论回溯**——在实践基础上系统讲授 MDP、贝尔曼方程、策略梯度定理、GAE 等理论，完成从直觉到形式化的闭环。
4. **前沿衔接与 LLM 时代**——深入大模型后训练（Post-Training）的核心地带。从 RLHF 三阶段完整流水线、离线对齐方法族（DPO / KTO / SimPO）的理论等价性，到 Pure-RL 范式下的无 Critic 算法（GRPO / DAPO）与可验证奖励（RLVR），全面解析大语言模型的强化学习实践。
5. **多模态与 Agentic RL**——跨越单轮文本交互的边界。探讨 VLM（视觉语言模型）的强化学习挑战与奖励分配；深入 Agentic RL，拆解多轮工具调用、Web Agent / Code Agent 的轨迹合成机制、信用分配难题（Credit Assignment）以及 Deep Research Agent 的前沿工程落地，揭示从基础对齐到自主智能体进化的完整路径。

## 课程目标

学完本课程后，学习者应能够：

- 理解强化学习的核心数学框架（MDP、价值函数、策略梯度），并能够用代码将其实现。
- 阅读并理解 DQN、PPO、DPO、GRPO 等主流算法的原始论文及其工程实现。
- **精通 LLM 强化学习后训练（Post-Training）**：熟练掌握 RLHF 完整流水线与数据工程（SFT / RM / PPO），理解 DPO 等离线对齐方法的数学本质，并能运用 GRPO / DAPO 与可验证奖励（RLVR）范式驱动模型涌现推理能力（如 DeepSeek-R1-Zero 范式）。
- **具备 Agentic RL 与 VLM 的前沿工程实战能力**：能够应对多轮交互中的信用分配挑战，构建轨迹合成与自进化数据飞轮，实现具备复杂工具调用（Tool Use）能力的 Web Agent 与 Code Agent，并掌握视觉语言模型的特殊奖励分配技巧。
- 针对给定的实际问题，合理选择并调试适用的强化学习算法，具备解决策略崩溃、奖励投机等实际工程问题的经验。

## 课程大纲

### 前言

| 课题                                                   | 说明                 |
| :----------------------------------------------------- | :------------------- |
| [写在开头](docs/preface/intro.md)                      | 课程定位与学习路线。 |
| [强化学习简史](docs/preface/brief-history/index.md)    | 从试错学习到 AlphaGo 到 LLM 对齐。 |

### 第一篇：基础导论

| 章节           | 课题                                                                 | 核心内容                                                                                                             |
| :------------- | :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Chapter 01** | [RL 初印象：求解 CartPole](docs/chapter01_cartpole/intro.md)         | 运行第一个 CartPole 训练脚本，理解核心原理；观察训练与指标变化，在实验中理解状态、动作、奖励、策略等基本要素的含义。 |
| **Chapter 02** | [现代 RL 初体验：大语言模型与 DPO 对齐](docs/chapter02_dpo/intro.md) | 用 DPO 算法对大语言模型进行偏好微调，直观体验后训练（post-training）的完整流程与核心原理；分析训练与指标表现。       |

### 第二篇：核心理论与方法

| 章节           | 课题                                                               | 核心内容                                                                                                                                                                 |
| :------------- | :----------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chapter 03** | [RL 理论基石：状态、动作与价值](docs/chapter03_mdp/intro.md)       | 动手实践两台老虎机（Bandit）；引入 MDP 形式化定义与价值函数；推导贝尔曼方程，建立对 TD Error 的直觉；了解经典方法与强化学习路线图。                                      |
| **Chapter 04** | [深度强化学习 DQN](docs/chapter04_dqn/intro.md)                    | 从经典 Q-Learning 到 DQN 的演进；拆解 DQN 三大组件；通过 CartPole、Atari 像素级输入、ViZDoom 及 stable-retro 宝可梦实战检验算法；观察训练过程，解析 DQN 家族与视角迁移。 |
| **Chapter 05** | [策略梯度与 Actor-Critic](docs/chapter05_policy_gradient/intro.md) | 从摇骰子赌博机实验出发，推导策略梯度定理与 REINFORCE；引入基线构建 Actor-Critic 架构；完成基线实验总结与 AlphaGo 的简单复现。                                            |
| **Chapter 06** | [PPO 与奖励模型](docs/chapter06_ppo/intro.md)                      | 动手训练 LunarLander；深入剖析 PPO 数学推导、信任域与裁剪机制；理解 GAE 的优势估计、奖励模型机制及其在 LLM 对齐中的作用。                                                |

### 第三篇：大模型强化学习

| 章节           | 课题                                                               | 核心内容                                                                                                                                                              |
| :------------- | :----------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chapter 07** | [对齐方法族：DPO / KTO / SimPO](docs/chapter07_alignment/intro.md) | 动手实践 DPO 对齐实验；推导 DPO 的数学本质与隐式奖励模型；对比 DPO 家族不同衍生方法的原理，提供在实际场景中的选型指南。                                               |
| **Chapter 08** | [GRPO、DAPO 与 RLVR](docs/chapter08_grpo_rlvr/intro.md)            | 动手用 GRPO 训练数学推理能力；剖析 GRPO 核心机制；探讨 DeepSeek 范式、DAPO 与 RLVR；展望 RL Scaling 与测试时计算前沿；深入知识蒸馏与在线策略蒸馏。                   |

### 第四篇：前沿与进阶专题

| 章节           | 课题                                                                 | 核心内容                                                                                                                                                                                         |
| :------------- | :------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chapter 09** | [连续动作控制 (SAC/TD3)](docs/chapter09_continuous_control/intro.md) | 在 PyBullet 机器人仿真中体验连续控制；解析连续策略与 DDPG/TD3；对比 SAC 算法与并行采样技术；引入 HER 解决稀疏奖励问题；探讨扩散策略在生成式连续控制中的应用。                                    |
| **Chapter 10** | [RLHF 完整流水线](docs/chapter10_rlhf/intro.md)                      | 全景展示模仿学习与数据工程流水线；探讨奖励函数设计原则；应对训练稳定性挑战与防范奖励黑客；前瞻 RLAIF 与自我博弈机制；构建数据循环体系。                                                           |
| **Chapter 11** | [VLM 强化学习](docs/chapter11_vlm_rl/intro.md)                       | 动手使用 GRPO 训练 VLM 回答视觉问题；讨论视觉与文本特征融合下的特殊挑战与奖励分配；介绍主流 VLM RL 框架与前沿进展；探讨视觉生成模型的 RL 后训练。                                                  |
| **Chapter 12** | [Agentic RL](docs/chapter12_agentic_rl/intro.md)                     | 从单轮交互扩展到多轮交互 RL 与信用分配；解析轨迹合成与数据工程；实战工具调用 RL（Web Agent 与 Code Agent）；梳理工业界 Agentic RL 实践、评测体系与 Benchmark 全景；深入 Deep Research Agent。    |
| **Chapter 13** | [未来趋势](docs/chapter13_future_trends/intro.md)                    | 测试时计算与 RL 推理；多模态与具身智能融合；多智能体 RL 与基于模型的 RL；LLM 多智能体强化学习；自博弈与自进化学习路线；离线强化学习回顾（CQL / IQL / DT）；实战 PettingZoo 多智能体环境。         |

### 附录

| 附录           | 课题                                                           | 说明                                                                                                                                                     |
| :------------- | :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Appendix A** | [强化学习训练调试指南](docs/appendix_common_pitfalls/intro.md) | 针对策略崩溃、奖励投机、资源溢出与收敛失效等常见故障，提供现象描述、理论解释与修复验证方法。                                                             |
| **Appendix B** | [RL 工程实践指南](docs/appendix_industrial_training/intro.md)  | 涵盖 RL 采样基础设施、异步训练架构、分布式并行策略、Agentic RL 基础设施、评测体系与 Badcase 分析、训练监控与排查、工业实战练习及大模型 RL 训练指标词典。 |
| **Appendix C** | [算法选型与工程框架](docs/appendix_algorithm_guide/intro.md)   | 算法选型决策矩阵与训练框架（含模型基方法）选型指南。                                                                                                     |
| **Appendix D** | [强化学习经典项目](docs/appendix_game_projects/intro.md)       | 拓展视野的开源强化学习项目列表。                                                                                                                         |
| **Appendix E** | [数学基础](docs/appendix_math/intro.md)                        | 涵盖线性代数、概率与统计、微积分与优化、信息论等强化学习所需的数学基础。                                                                                 |
| **Appendix G** | [术语对照表](docs/appendix_terminology/intro.md)               | 中英文强化学习术语对照及简明释义。                                                                                                                       |
| **Appendix H** | [环境安装指南](docs/appendix_env_install/intro.md)             | 课程相关工具与环境的安装配置说明。                                                                                                                       |

## 本地运行环境

本课程的文档站点基于 [VitePress](https://vitepress.dev/) 构建。在本地预览或贡献内容，需要以下步骤：

### 前置要求

- Node.js >= 18.0.0

### 安装与启动

```bash
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

启动后访问 `http://localhost:5173` 即可浏览课程内容。

### 常用命令

```bash
npm run build    # 构建静态站点
npm run preview  # 本地预览构建产物
npm run verify   # 构建检查（提交 PR 前请务必执行）
npm run format   # 代码格式化
npm run lint     # 代码规范检查
```

## 仓库结构

```text
hands-on-modern-rl/
├── docs/                      # 课程文档（VitePress）
│   ├── .vitepress/            # 站点配置与自定义组件
│   ├── public/                # 静态资源
│   ├── preface/               # 前言与导论
│   ├── chapter*/              # 各章节 Markdown 文件
│   ├── appendix*/             # 附录与补充材料
│   └── summaries/             # 各篇总结
├── scripts/                   # 自动化脚本（站点地图生成等）
├── .github/workflows/         # CI/CD 配置
├── package.json               # 项目配置
└── AGENTS.md                  # 仓库维护规则
```

## 参与贡献

欢迎通过 Pull Request 或 Issue 参与本课程的改进。为保证协作效率，请注意以下几点：

1. **保持单一职责**：每个 PR 聚焦于一个明确的改动，避免混合不相关的修改。
2. **遵循目录命名规范**：在 `docs/` 下新增内容时，使用连字符命名法（如 `chapter16-new-topic`），并以 `index.md` 作为入口文件。
3. **提交前验证**：若修改涉及 `.vitepress/config.mjs` 或构建脚本，请运行 `npm run verify` 确认无破坏性变更。
4. **Commit 规范**：遵循 [Conventional Commits](https://www.conventionalcommits.org/)（如 `feat:`, `fix:`, `docs:`, `chore:`）。

详细的维护与协作规则参见 [AGENTS.md](./AGENTS.md)。

## 许可协议

本课程内容基于 [CC BY-NC-SA 4.0（署名-非商业性使用-相同方式共享 4.0 国际）](./LICENSE) 协议发布。

---

<div align="center">
  <sub>Maintained by WalkingLabs and the open-source community.</sub>
</div>
