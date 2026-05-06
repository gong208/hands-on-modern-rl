# 9.7 工业界后训练实践全景

把 DPO、GRPO、RLVR 放到真实公司里看，会发现后训练已经不是一个单独算法，而是一整套生产系统：数据合成、SFT、偏好优化、可验证奖励、在线 rollout、工具环境、评测、拒答和安全策略会混在一起迭代。下面按公开资料梳理截至 2026-05-05 能查到的主流做法。这里不做排行榜；如果某家公司只公开了发布会或模型卡级别的信息，我会明确标注“披露有限”，避免把营销表述误写成训练 recipe。

## 国内大厂与主流实验室

### MiniMax

- **核心报告/模型**：[^minimax_m2_1]、[^minimax_m1]、[^minimax_webexplorer]
- **后训练重点**：
  - **可验证环境构建**：将 Agent 后训练拆解为“可验证环境 + 多 scaffold SFT/RL”。例如 SWE Scaling 从 GitHub 构建 Docker 环境，AppDev 使用 Playwright 进行交互检查。
  - **多维奖励信号**：奖励不再局限于固定的测试用例，而是结合专家编写的 rubric 交互评分以及长程搜索轨迹（如 WebExplorer）。
- **能借鉴的地方（实践启示）**：
  - **跨框架泛化**：在 RL 阶段引入不同 scaffold、工具协议和环境噪声，目标是让模型学会泛化，而非仅仅适配某个特定的 ReAct 模板。
  - **Agent-as-a-Verifier**：利用工具自动执行验证，提升奖励信号的可靠性，降低人工标注成本。

### 阿里 Qwen / 通义

- **核心报告/模型**：[^qwen2_5]、[^qwen2_5_math]、[^qwq_32b]、[^qwen3]、[^qwen3_coder]、[^tongyi_dr]
- **后训练重点**：
  - **数学自改进到 QwQ**：Qwen2.5-Math 利用 self-improvement pipeline 构造训练信号。QwQ-32B 则是直接的 reasoning RL 宣言，在强底座上用 RL 放大推理能力。
  - **分阶段 RL（Qwen3）**：将后训练拆分为 Long-CoT 冷启动、推理 RL、思考/非思考模式融合、通用能力 RL。
  - **Agentic Coding 与 Deep Research**：Qwen3-Coder 面向工程环境训练策略。Tongyi DeepResearch 利用 token-level policy gradient 和 leave-one-out advantage，将模型从“会检索”推向“能长期规划并产出报告”。
- **能借鉴的地方（实践启示）**：
  - **分段能力回填**：纯推理 RL 容易牺牲通用对话能力。先练长链思考，再强化数学逻辑，最后用偏好和安全数据回填通用能力，是维持全能模型的最佳实践。
  - **屏蔽工具 token Loss**：在 tool-integrated reasoning 中，不计算工具返回结果的 loss，防止模型把外部工具的输出当作自己生成的推理。

### Moonshot Kimi

- **核心报告/模型**：[^kimi_k1_5]、[^kimi_k2]、[^kimi_researcher]
- **后训练重点**：
  - **RL Scaling 与长度控制（k1.5）**：使用在线 policy mirror descent，在 KL 约束下不断更新。同时用 length reward 抑制无意义的“过度思考”。
  - **Agentic RL 与多环境任务（K2）**：将真实和合成任务转化为多轮轨迹，利用 verifier 和 rubric judge 筛选。
  - **研究助理范式（Kimi-Researcher）**：不仅评判最终答案，还评估搜索路径、证据覆盖、引用可靠性及是否绕弯。
- **能借鉴的地方（实践启示）**：
  - **Long-to-short 压缩**：通过技术手段将长推理能力压缩到更短的输出中，提高推理效率。
  - **多维评估器**：结合经典 RM 和带推理能力的评估器，以处理不同形态的答案（如数学、代码、开放性问题）。

### 字节 Seed / Doubao

- **核心报告/模型**：[^seed1_5_thinking]、[^vapo]、[^dapo]、[^ui_tars]、[^ui_tars_2]、[^seed_prover]、[^seed1_8]
- **后训练重点**：
  - **RL 基础设施与工程食谱（DAPO/VAPO）**：DAPO 提供了大规模 RL 的开源方案。VAPO 解决 value-model-based RL 的 bias、KL 控制等核心问题。
  - **GUI Agent（UI-TARS）**：将后训练推向视觉与界面交互，在多端环境中接受环境反馈，强调 multi-turn RL 和 sandbox rollout。
  - **数学证明与泛化（Seed Prover/Seed1.8）**：结合形式化证明、搜索和 Agent 架构。
- **能借鉴的地方（实践启示）**：
  - **Token-level Policy Gradient**：在长推理链中，让每个有效 token 都产生梯度，解决超长输出和稀疏奖励问题。
  - **Dynamic Sampling & Clip-Higher**：过滤全对/全错样本，并防止低概率 token 过早被压死，提高 RL 的稳定性。

### DeepSeek

- **核心报告/模型**：[^deepseek_math]、[^deepseek_r1]、[^deepseek_v3_2]
- **后训练重点**：
  - **GRPO 前史（DeepSeekMath）**：提出 Group Relative Policy Optimization，用组内分数估计 baseline，省去 critic 模型，降低资源消耗。
  - **纯 RL 诱发推理（R1-Zero）**：证明无 SFT 冷启动，仅靠 GRPO 和规则奖励即可诱发反思和长 CoT。
  - **工程化 R1 与 V3.2**：R1 采用冷启动 -> Reasoning RL -> Rejection Sampling SFT -> 通用 RL 的四段式流水线。V3.2 进一步强调大规模 Agentic Task Synthesis，将 reasoning 融入工具使用。
- **能借鉴的地方（实践启示）**：
  - **组内相对优势（GRPO）**：相比 PPO 显著降低了显存开销和训练不稳定性，是当前开源界复现推理模型的主流选择。
  - **交互过程可验证**：后训练的目标从单一的“答案对错”扩展到“交互与工具调用过程是否正确”。

### 智谱 Z.ai / GLM

- **核心报告/模型**：[^glm_4_5]、[^glm_5]
- **后训练重点**：
  - **ARC 能力统一（GLM-4.5）**：Agentic, Reasoning, Coding 能力被统一，不再拆分训练。
  - **明确的分段 RL（GLM-5）**：依次进行 Reasoning RL -> Agentic RL -> General RL。
- **能借鉴的地方（实践启示）**：
  - **能力构建顺序**：先在数学/代码中建立长链推理能力，再去工具环境中学习行动，最后用通用偏好把体验拉回，这一顺序具有极强的工程指导意义。

### 腾讯 Hunyuan

- **核心报告/模型**：[^hunyuan_t1]、[^hunyuan_a13b]
- **后训练重点**：
  - **多阶段 SFT 与 RL（Hunyuan-A13B）**：支持快慢思考和长上下文，包含 outcome-based rewards、代码 sandbox 和工具反馈。
  - **角色合成与多步调用**：Agent 阶段合成 planner、checker、tool 等角色，让模型学习结构化推理。
- **能借鉴的地方（实践启示）**：
  - **多领域 RL 的融合**：把“可切换思考模式 + 多领域 SFT + reasoning/tool/agent RL”放在同一条后训练链路中，是构建全能 MoE 模型的标准实践。

### 百度 ERNIE

- **核心报告/模型**：[^ernie_4_5]、[^ernie_5_0]
- **后训练重点**：
  - **Progressive RL 与 UPO（ERNIE 4.5）**：使用 Progressive Reinforcement Learning，通过 Unified Preference Optimization 处理不同任务的 reward 分布差异。
  - **多模态统一后训练（ERNIE 5.0）**：在文本、图像、视频、语音上稳定进行大规模后训练，结合 RLVR 和 thinking 模式。
- **能借鉴的地方（实践启示）**：
  - **统一奖励系统（UPO）**：解决多任务混合训练时奖励尺度不一致的问题，确保不同领域的能力平衡提升。

### 阶跃 StepFun

- **核心报告/模型**：[^step3]、[^step3_vl_10b]、[^step_deepresearch]
- **后训练重点**：
  - **多模态与 PaCoRe（STEP3-VL-10B）**：提出 Parallel Coordinated Reasoning，协调感知、推理和回答。
  - **Agentic Pipeline（Step-DeepResearch）**：将 deep research 拆分为 agentic mid-training、SFT、RL 和评测环境。
- **能借鉴的地方（实践启示）**：
  - **长程过程验证**：不仅评估单题答案，更评估长时间搜集证据、筛选来源、组织论证的过程是否可靠。

### 美团 LongCat

- **核心报告/模型**：[^longcat_flash]
- **后训练重点**：
  - **多环境 RL 与 DORA 异步系统**：自动构建覆盖数十种工具的可执行环境图谱。使用 DORA 异步流式训练系统解决多环境 rollout 的长尾等待问题。
  - **噪声鲁棒训练**：系统注入工具失败、返回缺失、指令歧义等扰动。
- **能借鉴的地方（实践启示）**：
  - **异步 Rollout 架构**：在 Agentic RL 中，不同环境反馈时间差异巨大，PD 解耦和异步控制是提升 GPU 利用率的关键。
  - **应对不完美环境**：真实的 reward 不仅来自正确答案，还来自模型能否在不完美环境中稳住计划。

### 蚂蚁 Ling / Ring

- **核心报告/模型**：[^ling_1t]、[^ring_1t]
- **后训练重点**：
  - **深度思考与高效推理**：重点放在 thinking model 能力和长序列处理上（注：公开材料偏高层，算法细节披露有限）。

### 华为 Pangu

- **核心报告/模型**：[^pangu_ultra]、[^pangu_pro_moe]、[^pangu_news]
- **后训练重点**：
  - **昇腾原生与 MoE 稀疏效率**：强调基于 Ascend NPU 的大规模训练系统优化，以及通过 post-training 增强 reasoning 和快慢思考融合。

### 01.AI Yi

- **核心报告/模型**：[^yi_lightning]
- **后训练重点**：
  - **高质量数据与指令后训练**：偏向传统 LLM alignment（SFT、RLHF），重点在于把聊天、推理、代码、长上下文做扎实，而非大规模工具环境。

### InternLM / 上海 AI Lab

- **核心报告/模型**：[^internlm2]
- **后训练重点**：
  - **在线 RLHF（COOL）**：使用 Conditional Online RLHF 处理偏好优化，构建奖励模型和数据治理流程，减少能力漂移。
- **能借鉴的地方（实践启示）**：
  - **传统 RLHF 工程化**：为开源社区展示了如何构建数据治理和条件化训练，是偏好优化的标杆参考。

### 百川 Baichuan 与 360 智脑

- **核心报告/模型**：[^baichuan2]、[^zhinao]
- **后训练重点**：
  - **经典流水线**：Baichuan2 展示了 SFT -> RM -> PPO 的经典对齐流程。
  - **数据中心化（360 智脑）**：强调数据质量优于数量，利用 RM 作为 judge 和数据过滤器。
- **能借鉴的地方（实践启示）**：
  - **数据反刍与清洗**：后训练是反复过滤、重标、reweight 和用模型反馈修数据的过程。

### 昆仑万维 Skywork 与 小米 MiMo

- **核心报告/模型**：[^skywork_or1]、[^mimo]、[^mimo_vl]
- **后训练重点**：
  - **强蒸馏上的 RL（Skywork-OR1）**：在 R1-Distill 之上解决 entropy collapse 和训练稳定性问题，探索 7B/32B 尺度上的性能变化。
  - **小模型推理（MiMo）**：构造 130K 可验证数学和编程问题做 RL，使用难度驱动的奖励和战略性数据重采样。
- **能借鉴的地方（实践启示）**：
  - **防止坍缩**：在已经具备思考能力的模型上继续 RL 时，如何通过规则和采样策略让其继续探索而不坍缩。
  - **高质量验证数据**：小模型 reasoning 不只是蒸馏，高质量可验证数据和稳定 RL recipe 同样能带来显著提升。

### 快手、商汤、讯飞

- **核心报告/模型**：[^keye_vl]、[^sensenova_u1]、[^spark_x1]
- **说明**：这三家公司公开了多模态后训练（快手）、原生理解生成（商汤）和深度推理（讯飞）的动态，但缺乏完整的训练 recipe 报告，适合作为产业动态参考。

---

## 国外大厂与主流实验室

### OpenAI

- **核心报告/模型**：[^instructgpt]、[^gpt4]、[^o1]、[^o3_o4_mini]、[^o3_operator]、[^gpt4_5]、[^gpt5]、[^gpt5_1]、[^gpt5_4]、[^gpt5_5]、[^gpt5_codex]
- **后训练重点**：
  - **RLHF 奠基**：从 InstructGPT 的三段式到 GPT-4 的全面对齐。
  - **大规模 Reasoning 与 Tool RL**：o-series（o1/o3/o4-mini）将工具调用（浏览、代码、文件）融入 CoT 思考过程。
  - **Coding Agent 与安全评测**：如 GPT-5-Codex 将真实软件工程任务、代码执行纳入 RL 评测。持续通过系统卡披露安全、幻觉和越狱的评测框架。
- **能借鉴的地方（实践启示）**：
  - **内化工具调用**：模型不再是“先想完再调工具”，而是在推理链内部决定何时使用工具。
  - **Deliberative Alignment**：通过长时间思考，学会在安全策略上做上下文推理。

### Anthropic

- **核心报告/模型**：[^constitutional_ai]、[^anthropic_cai]、[^claude4]、[^claude_sonnet_4_5]、[^claude_opus_4_5]、[^claude_opus_4_6]
- **后训练重点**：
  - **Constitutional AI**：通过设定原则，让 AI 自我批改（生成 SFT 数据）和对比（训练 PM），最终进行 RLAIF。
  - **Frontier Alignment 评测**：Claude 4 及其后续系统卡重点公开了针对 reward hacking、sabotage、sycophancy 和 jailbreak 的深度评测框架。
- **能借鉴的地方（实践启示）**：
  - **AI Feedback 的系统化**：大规模安全数据难以全靠人工，利用规则、模型裁判和红队评测协同工作是必由之路。
  - **反向证明机制**：强调“训练后如何证明模型没有学到危险目标”，这对安全后训练至关重要。

### Google DeepMind

- **核心报告/模型**：[^gemini_1_5]、[^gemini_2_5]、[^gemini_2_5_deep_think]、[^gemini_2_5_computer_use]、[^gemini_3_1_pro]、[^gemma_3]
- **后训练重点**：
  - **多模态与长上下文 Agent**：Gemini 系列强调百万级上下文、tool use 和 agentic workflow。
  - **Deep Think 与 Computer Use**：Gemini 2.5 Deep Think 训练模型在多条候选思路间探索整合；Computer Use 模型直接处理 GUI 状态和多步交互任务。
  - **开源提效（Gemma 3）**：展示了 distillation 和新型 post-training recipe 对小模型能力的显著提升。
- **能借鉴的地方（实践启示）**：
  - **探索与整合并重**：Frontier reasoning 已经超越了输出长 CoT，重点在于训练模型如何生成多条思路并进行有效的综合比较。

### Meta Llama

- **核心报告/模型**：[^llama3_herd]
- **后训练重点**：
  - **最完整的开放模型 Post-training 参考**：详尽拆解了 SFT、拒绝采样、奖励模型、偏好优化、安全数据和红队评测的协同工作。
- **能借鉴的地方（实践启示）**：
  - **评测闭环与数据配方**：Meta 的成功在于不同能力域匹配不同的 SFT 数据，用 RM 筛选排序，用偏好优化提升体验，这是一套标准的产品级 chat model 工业生产线。

### Microsoft Phi

- **核心报告/模型**：[^phi_4]、[^phi_4_reasoning]
- **后训练重点**：
  - **高质量合成数据与短 RL**：在精挑细选的 teachable prompts 上做 SFT，再叠加一段 outcome-based RL，使模型生成更长、更有效的推理轨迹。
- **能借鉴的地方（实践启示）**：
  - **小模型的经济账**：小模型无需巨大的 RL 预算，把可教数据和难度分布设计好，再用短 RL 修正输出长度和正确率，性价比极高。

### NVIDIA Nemotron

- **核心报告/模型**：[^nemotron_4]、[^llama_nemotron]、[^nemotron_ultra]、[^nemotron_agents]、[^nemotron_h]、[^nemotron_3]
- **后训练重点**：
  - **可复用后训练资产**：Nemotron-4 提供了数据生成、偏好数据和 RM 的完整组件。
  - **Curriculum-driven RLVR（Llama-Nemotron）**：仅靠蒸馏无法超越 teacher，必须依赖 curriculum-driven RLVR。
  - **Hybrid Architecture（Nemotron-H）**：先打磨长 CoT/STEM，再混入指令跟随和安全数据，服务企业 agent workflow。
- **能借鉴的地方（实践启示）**：
  - **生态打包**：不仅发布模型，更是将数据、RM、评测和 NIM 部署引擎打包，提供企业级解决方案。

### Mistral

- **核心报告/模型**：[^magistral]
- **后训练重点**：
  - **纯 RL Reasoning**：Magistral 明确表示使用自家的 scalable RL pipeline，而非蒸馏已有模型的 traces。
- **能借鉴的地方（实践启示）**：
  - **通用能力的保持**：实验表明，即使只在文本数据上做纯 RL，依然能够维持甚至提升多模态理解、指令跟随和函数调用能力。

### Apple

- **核心报告/模型**：[^apple_fm]、[^apple_fm_2025]
- **后训练重点**：
  - **端侧约束下的 RLHF**：SFT 混合人工与合成数据；RLHF 使用分布式异步 infrastructure（分离 policy updater 和 trajectory generators）。
  - **多源奖励**：包括文本/图文偏好 RM、数学与 STEM 规则验证。
- **能借鉴的地方（实践启示）**：
  - **工程化与隐私**：把后训练与端侧设备约束、隐私保护和多语言安全统一设计，是部署消费级 AI 的范本。

### xAI Grok

- **核心报告/模型**：[^grok_1]、[^grok_4]、[^grok_4_1]、[^grok_4_1_card]
- **后训练重点**：
  - **RL Scaling 与人格对齐**：使用 Colossus 进行接近预训练规模的 RL 提升推理。同时优化 personality、style、emotional intelligence 和 truthfulness。
- **能借鉴的地方（实践启示）**：
  - **产品级目标融入 RL**：人格、风格和帮助性也可以作为 RL 的优化目标，这是从技术模型走向 C 端产品的关键一步。

### IBM Granite

- **核心报告/模型**：[^granite_3_3]、[^granite_4_0]、[^granite_4_1]
- **后训练重点**：
  - **企业小模型 Reasoning**：通过 TPO 和 GRPO 提升复杂数学推理，支持可切换 thinking 模式。
  - **平衡与低成本**：围绕 RAG、工具调用、安全和低成本推理做平衡，采用 model merging 融合专家能力。
- **能借鉴的地方（实践启示）**：
  - **GRPO 的普及化**：Reasoning RL 不再是千亿参数闭源模型的专利，企业级小模型同样能利用 GRPO 强化特定领域能力。

### Salesforce xLAM / SFR-RL

- **核心报告/模型**：[^xlam]、[^sfr_rl]
- **后训练重点**：
  - **Action Model 与 RL 基础设施**：xLAM 专注于 API 和工具环境的动作选择。SFR-RL 提出 pipelined synchronous RL，解决长程工具调用中 rollout 长度差异导致的同步 RL 空等问题。
- **能借鉴的地方（实践启示）**：
  - **系统级优化**：全集群在 rollout 和 training 间切换，在保持 on-policy 更新的同时极大提高了 GPU 利用率。

### Amazon Nova

- **核心报告/模型**：[^nova]、[^nova_report]、[^nova_premier]、[^nova_forge]
- **后训练重点**：
  - **Nova Forge 企业 RL 平台**：允许企业客户混合私有数据，通过远程 reward functions 对齐模型。
- **能借鉴的地方（实践启示）**：
  - **Reward As a Service**：奖励不再局限于简单的 Python 脚本，而是扩展为企业内部系统 API、物理仿真或代码评测，展示了“后训练即服务”的产业化形态。

### Cohere Command A

- **核心报告/模型**：[^cohere_research]、[^command_a]
- **后训练重点**：
  - **Decentralized Pipeline**：核心 SFT 后，并行训练 code, safety, math 等多个 expert track，最后通过参数合并汇总能力。
  - **多样化偏好阶段**：包含 offline preference, online RL, best-of-N supervised training 和 RL Soup。
- **能借鉴的地方（实践启示）**：
  - **并行训练与合并**：企业模型的后训练可以是非线性的，通过并行训练多专家并利用 merge 技术，能大幅缩短研发周期并降低能力冲突。

### Databricks, AI21, Cursor, LG, NAVER, AI2 Tulu 3

- **核心报告/模型**：[^dbrx]、[^jamba_1_5a]、[^jamba_whitepaper]、[^cursor_composer_2]、[^exaone_4_0]、[^k_exaone]、[^hyperclova_x]、[^hyperclova_x_think]、[^tulu_3]、[^tulu_3_blog]、[^rl_survey]
- **补充视角与参考**：
  - **企业二次对齐（AI21 Jamba）**：通过合成安全偏好数据把企业 code of conduct 写入模型。
  - **Coding Agent 后训练（Cursor Composer 2）**：结合真实工程任务训练模型在代码库中的行动能力。
  - **本土化与混合偏好（LG/NAVER）**：韩国大厂展示了如何结合 SimPER/GROUPER 处理推理与非推理模式的融合，并兼顾本土文化。
  - **开放标杆（AI2 Tulu 3）**：完整开源了数据、代码和包含 SFT、DPO、RLVR 的配方，是学习如何组织一个完整开放 post-training 项目的最佳教科书。

---

## 读这些资料时抓四条主线

1. **奖励从“人喜欢哪个回答”变成“任务过程是否真的完成”。** 早期 RLHF 看 preference pair；R1、Qwen、Seed、Mistral 看答案可验证；MiniMax、Kimi、LongCat、Tongyi 看工具轨迹、环境状态和最终交付。
2. **数据从静态样本变成可生成、可验证、可回放的环境。** GitHub PR、Docker、Playwright、浏览器、数据库、工具图谱、搜索网页都变成后训练数据的一部分。
3. **后训练顺序越来越分段。** 常见顺序是 cold-start SFT、reasoning RL、agentic RL、general preference / safety 回填；顺序错了就容易出现长 CoT 过度、聊天退化、工具滥用或安全漂移。
4. **训练系统正在成为竞争力。** 异步 rollout、PD 解耦、KV-cache 交换、环境调度、失败恢复、reward service、LLM-as-judge 和可执行 verifier，都是“后训练实践”的一部分，而不是外围工程。

## 参考资料

[^minimax_m2_1]: [MiniMax M2.1: Post-Training Experience and Insights for Agent Models](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)

[^minimax_m1]: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)

[^minimax_webexplorer]: [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](https://arxiv.org/abs/2509.06501)

[^qwen2_5]: [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

[^qwen2_5_math]: [Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement](https://arxiv.org/abs/2409.12122)

[^qwq_32b]: [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)

[^qwen3]: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

[^qwen3_coder]: [Qwen3-Coder: Agentic Coding in the World](https://qwenlm.github.io/blog/qwen3-coder/)

[^qwen3_coder_next]: [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)

[^tongyi_dr]: [Tongyi DeepResearch Technical Report](https://arxiv.org/abs/2510.24701)

[^kimi_k1_5]: [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)

[^kimi_k2]: [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)

[^kimi_researcher]: [Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)

[^seed1_5_thinking]: [Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.13914)

[^vapo]: [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)

[^dapo]: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)

[^dapo_github]: [DAPO GitHub Repository](https://github.com/BytedTsinghua-SIA/DAPO)

[^seed1_5_vl]: [Seed1.5-VL Technical Report](https://arxiv.org/abs/2505.07062)

[^ui_tars]: [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326)

[^ui_tars_github]: [UI-TARS GitHub Repository](https://github.com/bytedance/ui-tars)

[^ui_tars_2]: [UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://huggingface.co/papers/2509.02544)

[^seed_prover]: [Seed Prover 1.5: Advanced Mathematical Reasoning through a Novel Agentic Architecture](https://seed.bytedance.com/en/blog/seed-prover-1-5-advanced-mathematical-reasoning-through-a-novel-agentic-architecture)

[^seed1_8]: [Official Release of Seed1.8: A Generalized Agentic Model](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model)

[^deepseek_math]: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

[^deepseek_r1]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

[^deepseek_v3_2]: [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)

[^glm_4_5]: [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)

[^glm_5]: [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/html/2602.15763v1)

[^hunyuan_t1]: [Hunyuan-T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)

[^hunyuan_a13b_instruct]: [Hunyuan-A13B-Instruct Model Card](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)

[^hunyuan_a13b]: [Hunyuan-A13B Technical Report](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)

[^ernie_4_5_family]: [ERNIE 4.5 Model Family](https://ernie.baidu.com/blog/posts/ernie4.5/)

[^ernie_4_5]: [ERNIE 4.5 Technical Report](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)

[^ernie_5_0]: [ERNIE 5.0 Technical Report](https://arxiv.org/abs/2602.04705)

[^step3]: [Step3: Cost-Effective Multimodal Intelligence](https://stepfun.ai/research/en/step3)

[^step3_vl_10b]: [STEP3-VL-10B Technical Report](https://huggingface.co/papers/2601.09668)

[^step_deepresearch]: [Step-DeepResearch Technical Report](https://arxiv.org/abs/2512.20491)

[^longcat_flash]: [LongCat-Flash-Thinking-2601 技术报告](https://tech.meituan.com/2026/02/02/longcat-flash-thinking-2601-techreport.html)

[^ling_1t]: [Ling-1T Model](https://ant-ling.medium.com/deep-insight-efficient-inference-introducing-the-trillion-parameter-ling-1t-model-77d6170e5e8e)

[^ring_1t]: [Ring-1T](https://ant-ling.medium.com/ring-1t-release-the-flow-state-of-insight-born-of-epiphany-c20e8e32817c)

[^pangu_ultra]: [Pangu Ultra](https://github.com/pangu-tech/pangu-ultra)

[^pangu_pro_moe]: [Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity](https://arxiv.org/abs/2505.21411)

[^pangu_news]: [华为宣布开源盘古 7B 稠密和 72B 混合专家模型](https://www.huawei.com/cn/news/2025/7/pangu-opensource)

[^yi_lightning]: [Yi-Lightning Technical Report](https://arxiv.org/abs/2412.01253)

[^internlm2]: [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297)

[^baichuan2]: [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)

[^zhinao]: [360Zhinao Technical Report](https://arxiv.org/abs/2405.13386)

[^skywork_or1]: [Skywork Open Reasoner 1 Technical Report](https://huggingface.co/papers/2505.22312)

[^skywork_or1_github]: [Skywork-OR1 GitHub Repository](https://github.com/SkyworkAI/Skywork-OR1)

[^keye_vl]: [Kwai Keye-VL Technical Report](https://arxiv.org/abs/2507.01949)

[^mimo]: [MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608)

[^mimo_github]: [Xiaomi MiMo GitHub Repository](https://github.com/XiaomiMiMo/MiMo)

[^mimo_vl]: [Xiaomi MiMo-VL-Miloco Technical Report](https://arxiv.org/abs/2512.17436)

[^sensenova_u1]: [SenseNova U1](https://www.sensetime.com/en/news-detail/51170629?categoryId=1072)

[^spark_x1]: [Spark X1 deep reasoning model](https://news.cgtn.com/news/2025-01-15/China-releases-Spark-X1-deep-reasoning-model-that-packs-a-punch-1AbIq8PzzEI/index.html)

[^instructgpt]: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

[^gpt4]: [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

[^o1]: [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)

[^o3_o4_mini]: [OpenAI o3 and o4-mini System Card](https://openai.com/index/o3-o4-mini-system-card/)

[^o3_operator]: [Addendum to o3 and o4-mini system card: OpenAI o3 Operator](https://openai.com/index/o3-o4-mini-system-card-addendum-operator-o3/)

[^gpt4_5]: [OpenAI GPT-4.5 System Card](https://openai.com/index/gpt-4-5-system-card/)

[^gpt5]: [OpenAI GPT-5 System Card](https://openai.com/index/gpt-5-system-card/)

[^gpt5_1]: [Addendum to GPT-5 system card: GPT-5.1](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1/)

[^gpt5_4]: [OpenAI GPT-5.4 Thinking System Card](https://openai.com/index/gpt-5-4-thinking-system-card/)

[^gpt5_5]: [OpenAI GPT-5.5 System Card](https://openai.com/index/gpt-5-5-system-card/)

[^gpt5_codex]: [Addendum to GPT-5 system card: GPT-5-Codex](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex/)

[^constitutional_ai]: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

[^anthropic_cai]: [Anthropic Constitutional AI overview](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)

[^claude4]: [System Card: Claude Opus 4 & Claude Sonnet 4](https://www.anthropic.com/claude-4-system-card)

[^claude_sonnet_4_5]: [Claude Sonnet 4.5 System Card](https://www.anthropic.com/claude-sonnet-4-5-system-card)

[^claude_opus_4_5]: [Claude Opus 4.5 System Card](https://www.anthropic.com/claude-opus-4-5-system-card)

[^claude_opus_4_6]: [Claude Opus 4.6 System Card](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf)

[^gemini_1_5]: [Gemini 1.5 Technical Report](https://arxiv.org/abs/2403.05530)

[^gemini_2_5]: [Gemini 2.5 Technical Report](https://arxiv.org/abs/2507.06261)

[^gemini_2_5_deep_think]: [Gemini 2.5 Deep Think](https://blog.google/products/gemini/gemini-2-5-deep-think)

[^gemini_2_5_computer_use]: [Gemini 2.5 Computer Use Model](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/)

[^gemini_3_1_pro]: [Gemini 3.1 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)

[^gemma_3]: [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)

[^llama3_herd]: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

[^phi_4]: [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)

[^phi_4_reasoning]: [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318)

[^nemotron_4]: [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704)

[^llama_nemotron]: [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949)

[^nemotron_ultra]: [NVIDIA Llama Nemotron Ultra Open Model](https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/)

[^nemotron_agents]: [Build Enterprise AI Agents with NVIDIA Llama Nemotron Reasoning Models](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)

[^nemotron_h]: [Nemotron-H Reasoning Model Family](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/)

[^nemotron_3]: [Inside NVIDIA Nemotron 3](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/)

[^magistral]: [Magistral](https://arxiv.org/abs/2506.10910)

[^apple_fm]: [Apple Intelligence Foundation Language Models](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models)

[^apple_fm_2025]: [Apple Intelligence Foundation Language Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)

[^grok_1]: [xAI Grok-1 Model Card](https://x.ai/news/grok/model-card)

[^grok_4]: [xAI Grok 4](https://x.ai/news/grok-4)

[^grok_4_1]: [xAI Grok 4.1](https://x.ai/news/grok-4-1/)

[^grok_4_1_card]: [xAI Grok 4.1 Model Card](https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf)

[^granite_3_3]: [IBM Granite 3.3](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras)

[^granite_4_0]: [IBM Granite 4.0](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)

[^granite_4_1]: [IBM Granite 4.1 Build Notes](https://huggingface.co/blog/ibm-granite/granite-4-1)

[^xlam]: [Salesforce xLAM](https://www.salesforce.com/blog/large-action-model-ai-agent/)

[^sfr_rl]: [Salesforce SFR-RL](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)

[^nova]: [Amazon Nova](https://aws.amazon.com/nova/)

[^nova_report]: [The Amazon Nova Family of Models: Technical Report and Model Card](https://www.isi.edu/results/publications/31887/the-amazon-nova-family-of-models-technical-report-and-model-card/)

[^nova_premier]: [Amazon Nova Premier: Technical report and model card](https://www.amazon.science/publications/amazon-nova-premier-technical-report-and-model-card)

[^nova_forge]: [Amazon Nova Forge](https://aws.amazon.com/nova/forge/)

[^cohere_research]: [Cohere Research](https://cohere.com/research)

[^command_a]: [Command A: An Enterprise-Ready Large Language Model](https://cohere.com/research/papers/command-a-technical-report.pdf)

[^dbrx]: [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)

[^jamba_1_5a]: [Jamba 1.5a: Enhancing AI Safety Through Post-Post-Training Alignment](https://www.ai21.com/research/jamba-1-5a/)

[^jamba_whitepaper]: [Jamba 1.5a Whitepaper](https://lp.ai21.com/hubfs/resources/Jamba-1-5a-Whitepaper.pdf)

[^cursor_composer_2]: [Cursor Composer 2 Technical Report](https://cursor.com/blog/composer-2-technical-report)

[^exaone_4_0]: [EXAONE 4.0 Technical Report](https://www.lgresearch.ai/data/cdn/upload/EXAONE_4_0.pdf)

[^k_exaone]: [K-EXAONE Technical Report](https://www.lgresearch.ai/data/cdn/upload/K-EXAONE_Technical_Report.pdf)

[^hyperclova_x]: [HyperCLOVA X Technical Report](https://arxiv.org/abs/2404.01954)

[^hyperclova_x_think]: [HyperCLOVA X THINK Technical Report](https://huggingface.co/papers/2506.22403)

[^tulu_3]: [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://openreview.net/forum?id=i1uGbfHHpH)

[^tulu_3_blog]: [Tulu 3 Technical Blog](https://allenai.org/blog/tulu-3-technical)

[^rl_survey]: [Reinforcement Learning for LLM Post-Training: A Survey](https://openreview.net/forum?id=UdsXTNzzvg)
