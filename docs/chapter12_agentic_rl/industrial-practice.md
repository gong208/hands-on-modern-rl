# 12.5 工业界实践：Agentic RL 训练中的常见问题与解决方案

前面几节介绍了 Agentic RL 的通用工程原则和框架设计。然而，在实际训练过程中，研究者往往会遇到一系列工程问题——训练不稳定、输出长度失控、奖励指标与实际质量脱节等。这些问题在学术论文中通常不会详细讨论，但对于工程实践至关重要。

2025–2026 年间，多家团队（包括 Alibaba、Moonshot、LinkedIn、Bespoke Labs 等）陆续公开了他们在 Agentic RL 训练中的实践经验。本节不再按团队逐一介绍，而是**按照实际训练中可能遇到的问题场景**进行组织，将不同团队的发现和解决方案汇总在一起。

> **核心要点**：在 Agentic RL 中，训练的稳定性往往比算法选择更为重要。数据质量和环境的一致性是决定训练效果的关键因素。

---

## 场景一：训练数据的获取与环境构建

许多研究者在开始 Agentic RL 训练时，首先面临的问题是：如何为模型提供稳定且可复现的交互环境？

### 真实 API 的局限性：不可复现性

如果直接接入真实的搜索引擎或代码执行环境进行训练，会遇到一个根本性的问题：**外部环境的输出是不可复现的**。

> **Moonshot AI** 在训练 Kimi-Researcher 时指出，Agent 所面对的环境是动态的——即使输入相同的查询，搜索引擎也可能返回不同的结果。他们在训练中主要采用了 **REINFORCE** 算法，并强调严格 On-policy 数据生成对训练稳定性的重要性 [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

### 合成环境的构建

一个可行的替代方案是构建确定性的合成环境，让模型在受控的条件下进行训练。

> **Alibaba 通义团队** 在构建 Tongyi DeepResearch 时，构建了一个以离线 Wikipedia 数据库和自定义工具为核心的合成训练环境，作为主要的训练平台。该合成环境提供快速且确定性的交互体验，有利于模型稳定学习。同时，他们也使用了冗余的备用搜索 API 等真实服务作为后备。实验表明，在合成环境中用合成数据训练的效果，可以优于在真实但噪声较大的标注数据上训练 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

### 小规模数据的有效性

对于资源有限的研究者而言，高质量的小规模数据同样可以取得显著效果。

> **Amazon Science** 在 AppWorld 基准测试上进行了验证：仅使用 **72 个高质量训练样本**进行 RL 训练，即可将 Qwen-2.5-32B 的任务完成率从 39.2% 提升至 72%，超过了 Claude Sonnet 3.7/4.0 的表现。这表明，高质量的数据结合模型在 RL 过程中的自主探索能力，能够以较小的数据规模实现有效的模型定制 [\[参考\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)。

---

## 场景二：训练初期的梯度爆炸问题

在解决了数据和环境的准备问题后，训练启动阶段的梯度爆炸是另一个常见问题。在排查超参数之前，应当首先检查底层实现的正确性。

### 推理引擎与训练引擎的实现差异

Agentic RL 的训练过程包含两个阶段：**推理（Rollout）** 阶段生成动作序列，**训练（Backward）** 阶段更新模型权重。这两个阶段通常由不同的引擎负责执行，而引擎之间的实现差异可能导致梯度计算不一致。

> **LinkedIn 团队** 在使用 GPT-OSS（一个 MoE 架构的开源模型）进行 RL 训练时，遇到了梯度爆炸和奖励不增长的问题。经过排查，他们发现根本原因是训练框架中 **Attention Sink 参数的反向传播未被实现**：推理引擎（SGLang 使用的 Triton kernel）支持 Attention Sink 的前向计算，但训练框架（FSDP 使用的 FlashAttention-v2）完全缺少对应的支持。他们从 vLLM 的 FlashAttention 分支中获取了前向实现，并自行编写了反向传播代码来计算 Sink 参数的梯度。修复该问题后，训练才恢复稳定 [\[参考\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)。

**实践建议**：在使用复杂模型架构时，建议先在简单的单轮任务（如 GSM8K）上验证训练流程的正确性，确认 Loss 正常下降后，再切换到多轮 Agent 任务。

---

## 场景三：输出长度失控与格式坍塌

这是 Agentic RL 训练中最常见的问题之一：模型未能学会正确使用工具，反而开始生成大量无意义的 token，最终退化为重复的乱码输出。这种现象被称为**格式坍塌（Format Collapse）**：

```json
// 期望的输出格式：
{"action": "search", "query": "AAPL stock"}

// 格式坍塌后的输出：
{"action": "searchsearchAAPL stockAAAAA"
```

下面分析导致这一问题的三个主要原因及其对应的解决方案。

### 原因一：奖励函数设计过于复杂

直觉上，研究者可能会设计多维度的奖励信号：工具调用成功给 +1，输出格式正确给 +1，最终答案正确给 +5。然而，这种细粒度的奖励设计可能适得其反。

**奖励博弈（Reward Hacking）** 是其中的核心问题。当奖励函数包含多个可被模型独立优化的子项时，模型可能找到只满足部分条件就能获得高奖励的策略。

> **Bespoke Labs** 的实验表明，包含工具调用次数奖励、格式检查奖励和正确性奖励的复合奖励函数，反而导致训练稳定性下降，推测原因正是奖励博弈。此外，他们还观察到输出长度持续膨胀、最终退化为无意义的乱码字符。他们最终采用的做法是：**仅保留"任务是否完成"这一个二值奖励信号**（通过 BFCL 的评估检查即为 1，否则为 0），删除所有中间过程的奖励项，训练稳定性反而显著提升 [\[参考\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)。

这一发现背后的逻辑是：二值的最终结果奖励不提供任何中间步骤的"捷径"，模型必须在整体上完成任务才能获得正向奖励，从而避免了针对单个奖励项的投机行为。此外，Bespoke Labs 还观察到复合奖励下输出长度持续膨胀并最终退化为乱码的现象，简化奖励设计后这一问题也得到了缓解。

### 原因二：负样本处理不当

在训练过程中，并非所有未能完成任务的样本质量都相同。例如，模型可能因为交互步数达到上限而被环境截断，此时并未产生最终答案，但在此之前的输出可能是合理的。如果将这类样本不加区分地作为负样本给予惩罚，可能会损害模型已经习得的输出能力。

> **Alibaba 通义团队** 观察到，不加过滤的负样本在长时间训练后会导致格式坍塌——模型逐渐丧失正确输出工具调用格式的能力。他们采用了**保守的负样本处理策略**：选择性地排除那些因超出长度限制而未产生最终答案的轨迹，使其不参与损失计算，从而保护模型的格式输出能力 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

### 原因三：KL 散度约束的配置不当

在 RLHF/GRPO 中，通常使用 KL 惩罚项来限制当前策略模型与初始参考模型之间的偏离程度。KL 约束的作用是防止策略在训练过程中偏离初始模型太远，从而维持输出的基本质量。

这一约束的配置需要在"允许策略探索"和"维持稳定性"之间取得平衡：

- **KL 惩罚过小**：约束力不足，策略可能偏离初始模型太远，导致输出质量退化。
- **KL 惩罚过大**：约束过强，策略难以学到新的行为，训练效果受限。

> **Bespoke Labs** 在训练 Qwen2.5-7B-Instruct 时发现，将 KL 惩罚设为 0 时，模型在约 300 步后即出现输出退化。他们采用的策略是：
>
> 1. **设置微小的 KL 权重**（如 0.001），提供最小程度的约束。
> 2. **定期更新参考模型**：每隔一定步数（如 100 步），将当前策略模型复制为新的参考模型。这样，KL 约束的目标会随训练推进而动态调整，避免策略被"锚定"在过远的初始状态上 [\[参考\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)。

### 输出长度控制：Gamma-decay 奖励

为了鼓励模型以更少的步数完成任务，可以引入基于步数衰减的奖励机制。

> **Moonshot** 提出了 **Gamma-decay Reward**。当模型正确完成任务时，奖励值随所用步数指数衰减：
>
> $$r_i = r \times \gamma^{T-i}, \quad \gamma < 1$$
>
> 其中 $T$ 是总步数，$i$ 是当前步数。这意味着：完成相同任务时，使用更少的步数会获得更高的奖励，从而引导模型学会更高效地执行任务 [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

---

## 场景四：长程交互中的上下文管理

Agentic RL 与传统 RL 的一个重要区别在于交互轮数可能非常长。在文献检索、代码编写、调试等复杂任务中，交互轮数可能超过 50 轮，此时上下文窗口会被大量历史信息填满，模型可能丢失对初始任务的关注。

### 上下文管理机制

> **Moonshot** 的 Kimi-Researcher 引入了 **上下文管理（Context Management）** 机制。其核心设计是通过一个 `context_manager` 函数，在每一轮交互中对当前状态进行筛选：保留重要信息，丢弃不必要的文档，从而将单次 Rollout 的交互轮次延长至 50 轮以上。早期消融实验显示，启用上下文管理后，模型平均多执行了 30% 的交互轮次，能够获取更多信息并取得更高的任务得分 [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

---

## 场景五：Agent 幻觉及其控制

在解决了训练稳定性和输出格式的问题之后，另一个需要关注的问题是 **Agent 幻觉（Hallucination）**：模型可能在搜索结果中引用不存在的文献，或者错误地使用 API 参数，却对后续推理表现出不恰当的"自信"。Agent 场景中的幻觉比纯对话场景更为复杂，因为模型不仅生成文本，还生成动作。

### Agent 幻觉的四种类型

**工具选择幻觉。** 模型调用了一个不存在的工具，或者在不该调用工具时强制调用。例如用户询问天气信息，模型却调用了 `execute_sql`。

**参数幻觉。** 工具选择正确，但参数填写错误——编造了不存在的 API 端点、拼错了数据库名、或使用了格式不正确的参数值。最值得警惕的是：参数格式可能看起来"合理"，但实际值是虚构的。

**结果幻觉。** 这是最隐蔽的幻觉类型。模型调用了正确的工具并获得了真实的返回结果，但在解读结果时引入了偏差——将搜索结果中的无关信息当作支持自己论点的证据，或忽略了与假设矛盾的内容。

**引用幻觉。** 模型声称"根据某文献/某网站"得出某个结论，但该引用实际上不存在，或引用内容与原文不符。这在 Deep Research Agent 中尤为常见——模型可能编造论文标题、URL 和统计数据来使输出"看起来有据可查"。

### Agent 幻觉的级联效应

在纯对话场景中，幻觉的后果通常限于提供错误信息。但在 Agent 场景中，幻觉会在多轮交互中**级联传播并自我强化**：

1. 第 3 轮：模型产生参数幻觉，调用了一个不存在的 API 参数 → 调用失败
2. 第 4 轮：模型未能识别幻觉，反而认为"该 API 存在缺陷" → 切换到另一个工具
3. 第 5 轮：新工具缺少关键功能 → 模型编造了一个看似合理的结论
4. 最终输出：一份表面上完整但建立在幻觉基础上的报告

更值得关注的是，如果 RL 的奖励仅基于最终输出质量（即 Outcome Reward），理论上模型可能发现"编造一个看似可信的答案"比"承认不确定"获得更高的奖励——这意味着 RL 训练反而可能**强化幻觉行为**。这一推断在逻辑上成立，但在已公开的工业界实践中尚未被明确报告为观察到的现象。

### RL 训练中的幻觉惩罚机制

**引用感知评分奖励。** 清华大学与智谱 AI 联合提出的 CaRR[^carr_industrial]（Citation-aware Rubric Rewards）设计了一种细粒度的奖励机制来引导模型正确引用证据。其核心思路是将多跳问题分解为一系列原子事实陈述（Rubrics），然后通过三步流程计算奖励：（1）检查模型输出是否识别了关键实体；（2）提取输出中引用的 URL，获取网页内容，判断每条 Rubric 是否被引用内容所支持；（3）通过图上的广度优先搜索验证各 Rubric 是否在逻辑上与最终答案相连通。最终奖励为被满足且逻辑连通的 Rubric 数量占总 Rubric 数量的比率。这一机制鼓励模型为每个论断提供可验证的、逻辑连贯的引用证据。

**工具结果忠实度奖励。** 鼓励模型在解读工具返回结果时忠实于原始内容。如果模型的总结与工具实际返回的信息存在偏差（通过 NLI 模型或交叉验证检测），则给予惩罚。

**不确定性奖励。** 鼓励模型在不确定时主动表达"需要更多信息"或"该结果不确定"，而非编造答案。综合上述三种策略，可以设计一个幻觉感知的奖励函数作为示例：

> **注意**：以下代码为说明性示例，综合了多种惩罚思路，并非直接来自某一篇论文的具体实现。

```python
def hallucination_aware_reward(answer, tool_results, citations):
    """幻觉感知的奖励函数"""
    reward = base_task_reward(answer)

    # 1. 引用真实性检查
    for citation in citations:
        if not verify_citation_exists(citation):
            reward -= 0.5  # 虚假引用，惩罚
        elif not verify_citation_supports(citation, answer):
            reward -= 0.3  # 引用与论断不符

    # 2. 工具结果忠实度
    for claim in extract_claims(answer):
        if has_supporting_evidence(claim, tool_results):
            reward += 0.1  # 有据可查的论断
        elif claim_is_verifiable(claim) and not has_supporting_evidence(claim, tool_results):
            reward -= 0.2  # 可验证但无证据的论断

    # 3. 鼓励不确定性表达（诚实奖励）
    if is_complex_question and ("不确定" in answer or "需要更多信息" in answer):
        if not all_claims_supported(answer, tool_results):
            reward += 0.15  # 在确实缺乏证据时，承认不确定性是合理行为

    return reward
```

### 基于验证的幻觉过滤

除了在奖励函数中惩罚幻觉外，还可以在**推理阶段**通过验证机制进行过滤：

**Self-RAG[^selfrag_industrial]** 提出了"自适应检索 + 自我评估"的框架。与传统 RAG 对每个查询都检索不同，Self-RAG 让模型在生成每个文本段**之前**，先通过特殊的反思 token（Reflection Token）判断是否需要检索外部信息。如果需要，则检索若干相关段落，为每条段落分别生成续写，并通过 [IsRel]（相关性）、[IsSup]（支撑度）、[IsUse]（有用性）等反思 token 对各候选续写进行打分，最终通过分段束搜索（Beam Search）选择综合得分最高的输出。该框架的核心特点是模型通过反思 token 实现了对自身输出的结构化自评估。

**CRITIC[^critic_industrial]** 提出了"工具辅助纠错"的幻觉过滤机制。模型生成初始回答后，主动调用外部工具（如搜索引擎、代码执行器）来验证关键论断，并基于工具反馈生成结构化的批评意见。如果批评意见表明回答存在问题，则模型基于批评意见重新生成修正后的回答。这一"验证→修正→验证"的循环可以迭代多轮，直到回答通过验证或达到最大迭代次数。与纯粹依赖模型自我评估的方法不同，CRITIC 引入了外部工具的客观反馈作为纠错依据。

### 幻觉控制实践总结

| 幻觉类型     | 检测方法                | RL 惩罚策略                 |
| ------------ | ----------------------- | --------------------------- |
| 工具选择幻觉 | 工具白名单校验          | 调用不存在工具 → reward = 0 |
| 参数幻觉     | Schema 校验 + 类型检查  | 参数格式错误 → 负向 reward  |
| 结果幻觉     | NLI 模型 + 交叉验证     | 论断与工具结果矛盾 → 惩罚   |
| 引用幻觉     | URL 可达性 + 内容相关性 | 虚假引用 → 惩罚             |

一个重要的实践原则是：**幻觉惩罚应在训练早期即引入**。一旦幻觉行为通过 RL 被强化，后续消除将非常困难。

---

## 场景六：特定模型架构的注意事项

前面的场景是大多数 Agentic RL 训练中都会遇到的共性问题。此外，使用特定的模型架构（如 MoE）或在较小参数量的模型上进行训练时，还可能出现一些额外的问题。

### MoE 模型的路由不确定性

MoE 模型（如 Mixtral、DeepSeek-V3）因推理成本较低而受到关注，但其路由机制可能破坏 RL 训练的基本假设。

PPO 等算法假设当前生成数据的模型与正在被训练的模型是同一个（即 On-policy），这在数学上表现为重要性采样比率等于 1。

> **LinkedIn 团队** 在使用 GPT-OSS 进行 RL 训练时发现，MoE 模型的路由网络（Gating Network）在两次前向传播中，可能为同一个 Token 选择不同的专家（Expert），导致 $\log \pi(a|s) \neq \log \pi_{\text{old}}(a|s)$，即 On-policy 假设被破坏。在排查过程中，他们曾尝试通过 `old_log_prob = log_prob.detach()` 的方式将两次概率强制对齐来验证这一假设。需要指出的是，该路由不一致问题虽真实存在，但在他们的调试中并非梯度爆炸的根本原因——根本原因在于上节所述的 Attention Sink 反向传播缺失 [\[参考\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)。

### MoE 模型的负载均衡问题

MoE 模型在 RL 训练中不仅面临上述路由一致性问题，还存在专家负载不均衡导致的 GPU 利用率低下。不同 Token 可能集中选择少数"热门"专家，导致负责这些专家的 GPU 成为瓶颈，而其他 GPU 则处于空闲状态。

> **Salesforce** 在其 SFR-RL 系统中提出了 **流水线同步 RL（Pipelined Synchronous）** 方案：所有 GPU 在 Rollout 和 Training 两个阶段之间交替切换，而非将 GPU 固定分配给某一阶段。此外，针对 MoE 模型，他们引入了 **Least-Loaded Expert Parallelism** 来优化专家的负载均衡。整体系统在内存效率上相比 VERL（FSDP + Context Parallelism）提升了约 250 倍，仅用 16 块 H200 即可训练 120B 参数的 MoE 模型 [\[参考\]](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)。

### 小模型的推理能力上限

需要注意的是，RL 的本质是**激发模型已有的能力**，而非注入新的知识。模型的基础能力决定了 RL 能够达到的效果上限。

> **Amazon Science** 的实验显示：32B 参数量的模型从 RL 中获益显著，因为模型本身能够生成高质量的交互轨迹（Rollout），形成正反馈循环。但较小的模型面临基础推理能力的限制，例如无法识别不可回答的问题或从相关上下文中提取答案——这种能力的缺失，RL 训练难以弥补。对于基础能力不足的小模型，研究者的建议是通过蒸馏（Distillation）从更强的模型中获取能力，而非单纯增加 RL 训练强度 [\[参考\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)。

### 分阶段训练管线

考虑到不同规模模型的特点，一个更稳健的训练策略是采用分阶段管线，而非直接进行 RL 训练。

> **Alibaba 通义团队** 设计了 **CPT → SFT → RL** 的三阶段训练管线：在预训练（CPT）阶段将工具调用的轨迹以文本形式融入训练数据；在 SFT 阶段培养模型的基本推理和工具使用能力；最后在 RL 阶段进行优化。他们发现，如果模型在起步阶段不具备基本的工具使用能力，直接进行 RL 训练往往效果不佳 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

---

## 实践总结 {#tricks}

下表汇总了各问题的对应解决方案：

| 问题                       | 解决方案                                                                     | 参考                  |
| -------------------------- | ---------------------------------------------------------------------------- | --------------------- |
| 训练环境不可复现           | 构建确定性的合成环境                                                          | Alibaba               |
| 小规模数据定制             | 高质量的小数据（如 72 条）结合 RL 也能取得显著效果                           | Amazon                |
| 训练初期梯度爆炸           | 检查推理引擎与训练引擎的底层实现一致性（如 Attention Sink 反向传播）           | LinkedIn              |
| 输出退化为重复乱码         | 采用极简奖励设计（仅奖惩任务成败）；对过长的输出进行过滤                      | Bespoke Labs          |
| 策略偏离初始模型           | 设置较小的 KL 惩罚（如 0.001）；定期将当前模型设为新的参考模型                | Bespoke Labs          |
| 输出效率低（步数过多）     | 使用 Gamma-decay 衰减奖励，鼓励以更少步数完成任务                             | Moonshot              |
| 格式坍塌                   | 采用保守的负样本处理策略，排除因超长截断而未产生最终答案的轨迹              | Alibaba               |
| 长任务上下文溢出           | 引入上下文管理机制，主动摘要或丢弃无用历史信息                                | Moonshot              |
| MoE 训练资源利用率低       | 流水线同步 RL + Expert Parallelism；16 块 H200 即可训练 120B MoE              | Salesforce            |
| MoE 路由不一致             | 注意 MoE 路由非确定性可能破坏 On-policy 假设；排查时需区分根因与表象          | LinkedIn              |
| 小模型训练效果不佳         | 通过蒸馏提升基础能力后再进行 RL；采用 CPT → SFT → RL 三阶段管线              | Amazon / Alibaba      |

## 参考资料 {#references}

- Zhu J, Sang H, et al. "[Unlocking Agentic RL Training for GPT-OSS: A Practical Retrospective](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)." Hugging Face Blog, 2026.
- Zhuang R, Vu T, et al. "[Improving Multi-Turn Tool Use with Reinforcement Learning](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)." Bespoke Labs Blog, 2025.
- Moonshot AI. "[Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)." 2025.
- Tongyi DeepResearch Team. "[Tongyi DeepResearch: From Chatbot to Autonomous Agent](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)." 2025. [GitHub](https://github.com/Alibaba-NLP/DeepResearch)
- Salesforce AI Research. "[Building Efficient RL Training for the Agentic Era](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)." 2026.
- Subramanian S, Xu P, Wang Y. "[Customizing Multiturn AI Agents with Reinforcement Learning](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)." Amazon Science Blog, 2026.

[^carr_industrial]: Zhang J, Lv X, Feng L, Hou L, Li J. "[Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards](https://arxiv.org/abs/2601.06021)." arXiv, 2026.

[^selfrag_industrial]: Asai A, et al. "[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)." ICLR 2024.

[^critic_industrial]: Gou Z, et al. "[CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)." ICLR 2024.

---

本节梳理了 Agentic RL 训练中的常见工程问题及工业界的解决方案。下一节将介绍这些技术的综合应用——[深度研究智能体：Deep Research Agent](./deep-research-agent)，展示 Agentic RL 如何训练能够自主进行长程研究的智能体。
