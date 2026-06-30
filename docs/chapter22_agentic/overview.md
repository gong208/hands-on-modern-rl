# 22.1 Agentic RL 总览

[本章导读](./intro) 给出了 Agentic RL 的形式化骨架——轨迹、POMDP、信用分配的几个公式。本节把这些公式落回到具体的工程图景：一个 Agent 在真实环境里如何运转、为什么训练它比训练一个单轮 LLM 难得多、当前工业界用什么框架把这件事跑起来。

读完本节，你应该能建立一个完整的"Agentic RL 训练系统"心智模型，知道后续每一节在哪一块上深入。

## 从单轮到多轮的范式转移

前面所有章节讨论的强化学习，本质上都是**单轮决策**：模型接收一个 prompt，输出一段完整回答，奖励模型给出一个分数，策略据此更新一次。无论底层算法是 PPO 还是 GRPO，"一问一答一打分"的骨架始终未变。

但真实的智能体不这样工作。

考虑一个订机票 Agent。用户说"帮我订一张明天北京到上海最便宜的早班机票"，Agent 必须分步行动：先搜索航班，对比价格和时间，确认座位库存，调用下单 API，等待出票确认。中间任何一步出错——搜索 query 太宽、没比价直接选第一条、库存判断失误、下单参数错误——整个任务就失败。环境只在最后给出一个二元信号：出票成功（reward = 1）或失败（reward = 0）。

这种从"一问一答"到"多步与环境交互"的转变，引出四个单轮 RL 完全没有的新挑战：

1. **训练对象从 completion 变成 trajectory**。一条轨迹混合了模型生成的 token、工具调用、工具返回、环境状态变化，结构上更像一棵对话树而非一段文本。
2. **Rollout 必须在真实环境里执行**。每一步都可能触发外部调用（搜索、API、代码执行），GPU 要等环境返回，利用率可能只有 20–30%。
3. **环境需要模块化、可复位、可验证**。训练时不能真去订一万张票，需要 sandbox 或模拟环境。
4. **多轮训练更容易失稳**。一条 10 步轨迹只有最后一步有奖励，前面的好决策与坏决策被同一个 reward 笼统评价——这就是**信用分配问题**。

本章围绕这四个挑战展开。先用两条对照轨迹建立直觉。

## 两条对照轨迹

同一个订机票任务，同一个模型，两次 rollout：

```
轨迹 A（成功）                          轨迹 B（失败）
─────────────────────────              ─────────────────────────
T1 search("北京 上海 早班 低价机票")     T1 search("北京 上海 机票")
   obs: 12 条相关航班                      obs: 200 条混合结果

T2 filter(dep<9:00, sort=price)         T2 pick_first()
   obs: CA1501 6:30 ¥760                   obs: MU5101 9:30 ¥1280

T3 check_seat(CA1501)                   T3 order(MU5101)
   obs: 座位充足                            obs: 下单成功

T4 order(CA1501, seat=window)
   obs: 出票成功

reward = 1                              reward = 0
```

两条轨迹最终 reward 截然不同，但**问题出在哪一步**？轨迹 B 失败是因为 T1 的 query 太宽，还是 T2 没比价直接选第一条，还是 T3 没确认就下单？只看最终 reward 无法回答。

- 形式化地把"轨迹 vs 单轮 completion"写清楚，见 [22.2 多轮 RL 形式化](./formulation)。
- 把最终 reward 回拆到每一步，见 [22.3 轨迹信用分配](./credit-assignment)。

## 智能体的基本组件

一个 agent 不只是 LLM。最小定义：**LLM 主干 + 指令 + 工具 + 环境**，四者在 agentic loop 中循环。

### LLM 主干

agent 的决策核心。接收当前观测，推理下一步，生成动作（文本或工具调用）。任何足够强的 LLM 都能充当主干，但工程上往往选用经过推理训练的模型——它们能在输出动作前先生成 thinking trace，对多步决策更友好。

### 指令

告诉 agent 要解决什么问题、用什么策略。除了任务本身（"订最便宜的早班机票"），还包含求解策略提示（"先搜索再筛选"、"价格和时间都要考虑"、"失败就重试"）。指令的质量直接决定 agent 行为的下限。

### 工具与环境

工具是 agent 操作环境的接口：搜索 API、代码解释器、CLI、MCP server、下单 API。工具调用通常以特殊 token 标记，嵌入到模型的 token 流中：

```
<tool_call>{"name":"search_flights","args":{"from":"PEK","to":"SHA"}}</tool_call>
<tool_response>[CA1501 6:30 ¥760, CA1831 7:00 ¥690, ...]</tool_response>
```

环境是有状态的：搜索结果会变化、库存会变动、下单会改变数据库。工具调用的返回不仅取决于参数，还取决于环境当前状态。这种把输出锚定到真实世界而非参数记忆的能力称为 **grounding**——是 agent 相对于纯 LLM 的一大优势，也是 RL 训练能赋予的核心行为模式。

### Agentic Loop

四者循环：**感知观测 → 推理下一步 → 执行动作 → 接收新观测**，直到满足终止条件（任务完成、达到最大步数、模型输出结束信号）。

一次完整的 loop 称为一次 **rollout**；rollout 产出的完整交互记录称为一条 **trajectory**，记作 $\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T)$。轨迹不只是文本序列，它混合了模型生成的 token、工具调用、工具返回、环境状态变化，结构上更像一棵对话树而非线性文本。

## 一个最简 Agent Loop

概念看十遍不如动手跑一遍。下面用几十行代码搭一个能跑的 Agent——不涉及 RL 训练，只看"一个 Agent 是怎么和工具交互的"。理解了这个循环，后面加上 RL 就水到渠成。

```python
import json, subprocess, os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

# ① 定义工具 与 告诉模型"你能做什么"
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command and return output",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

# ② 工具的实际执行逻辑（环境 Environment）
def execute_tool(name, args):
    if name == "execute_bash":
        r = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
        return r.stdout + r.stderr
    elif name == "read_file":
        with open(args["path"]) as f:
            return f.read()
    return f"Unknown tool: {name}"

# ③ Agent Loop 与 感知→推理→行动→观测，循环往复
def run_agent(task, max_turns=5):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": task},
    ]
    for turn in range(max_turns):
        # 感知 + 推理：模型根据当前信息决定下一步
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)

        # 行动：如果没有工具调用，说明模型给出了最终回答
        if not msg.tool_calls:
            return msg.content  # Agent 认为任务完成，退出循环

        # 观测：执行工具，把结果喂回给模型
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  [Turn {turn+1}] 调用工具: {tc.function.name}({args})")
            result = execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "（达到最大轮次，强制停止）"

# ④ 跑一下试试
print(run_agent("查看当前目录下有哪些 .md 文件，告诉我一共有几个"))
```

运行效果：

```
  [Turn 1] 调用工具: execute_bash({'command': 'ls *.md'})
  [Turn 2] 调用工具: execute_bash({'command': 'ls *.md | wc -l'})
当前目录下有 12 个 .md 文件。
```

把这 50 行代码和前面的概念对应起来：

- `tools = [...]` 对应**动作空间** $A_{\text{action}}$——定义了 Agent 可调用的工具。
- `execute_tool()` 对应**环境**——工具的实际执行逻辑。
- `for turn in range(max_turns)` 对应 **Agentic Loop / Rollout**——整个 for 循环就是一次完整的轨迹采样。
- `client.chat.completions.create()` 对应**策略 $\pi_\theta$**——模型决定下一步做什么。
- `messages.append(...)` 对应**状态 $s_t$**——整个对话历史就是当前状态。

这个 Agent 的"聪明程度"完全取决于策略 $\pi_\theta$。怎么给它加上 RL 训练——奖励怎么算（ORM vs PRM）、多步交互中奖励怎么分配（信用分配）、训练数据怎么管理——是后续小节要展开的。

## SFT 与 Prompting 的局限

一个自然的疑问：ReAct、Toolformer 等方法已经能让 LLM 调用工具了，为何还需要 RL？

关键区别在于：SFT 和 prompting 教会模型的是**模仿**——复制人类演示中"何时调用工具、调用什么工具"的模式。但真实的 Agent 任务中，工具使用的最优策略高度依赖上下文：

- 搜索查询如何构造？何时打开网页详情？何时停止搜索开始总结？
- 代码修改后测试仍未通过，是继续调试还是切换方向？
- 多个来源的信息相互矛盾，应采信哪一个？

这些本质上是**策略学习问题**，而非单纯的语言建模问题。演示数据难以覆盖所有可能的决策路径，而 RL 可以根据任务结果反向塑造工具调用、规划和记忆管理等行为模式。

SFT 和 RL 在 Agentic 场景中的分工：

- **SFT 教格式**：教会模型工具调用的语法（如调用搜索引擎的 JSON 格式）、基本的交互协议。
- **RL 教策略**：教会模型何时调用工具、如何组合多步行动、失败后如何恢复。

DeepSeek-R1-Zero 的实验表明，跳过 SFT 直接进行 RL 也能涌现出推理能力——但前提是基座模型足够强大。实践中，SFT warmup + RL fine-tuning 的两阶段方案仍是主流范式。

## 工业框架全景

回到现实——当你想真正训练一个 Agent 时，用什么框架把这套东西跑起来？

前面章节做 PPO、GRPO 时这个问题不尖锐：训练循环几乎全是 GPU 计算，TRL 或 OpenRLHF 都能轻松搞定。但 Agentic RL 的训练循环里多了一个"等"字——模型调用搜索引擎，GPU 就得等搜索结果；模型执行代码，GPU 就得等沙箱返回。怎么让 GPU 不空等？这就是 Agentic RL 训练框架要解决的核心问题。

2025–2026 年，围绕这个问题涌现了一批开源框架：

| 框架         | 开发方              | 一句话定位                                                        | 多轮 Agent 原生支持  | GitHub                                                    |
| ------------ | ------------------- | ----------------------------------------------------------------- | -------------------- | --------------------------------------------------------- |
| **OpenRLHF** | 开源社区            | 代码最简洁（8k 行），算法与 Agent 执行解耦，一行代码切换单轮/多轮 | 是                   | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |
| **verl**     | 字节跳动 / 开源社区 | 吞吐最高，训推在同一组 GPU 上动态切换，生态扩展最多               | 基础支持，社区扩展中 | [verl-project/verl](https://github.com/verl-project/verl) |
| **slime**    | THUDM / 智谱生态    | Megatron + SGLang 后训练框架，MoE 模型效率高                      | 基础支持             | [THUDM/slime](https://github.com/THUDM/slime)             |
| **AReaL**    | 蚂蚁 / 清华         | 全异步训练——GPU 完全不等，速度提升 2.77 倍                        | 是                   | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) |
| **ROLL**     | 阿里巴巴淘天        | 推理（RLVR）+ Agent 双模式，原生 Qwen 支持                        | 是                   | [alibaba/ROLL](https://github.com/alibaba/ROLL)           |
| **SkyRL**    | UC Berkeley         | 模块化全栈——训练、Agent 编排、任务环境各自独立                    | 是                   | [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)   |
| **Seer**     | Moonshot AI (Kimi)  | 极致同步——通过在线上下文学习消除 rollout 长尾，吞吐提升 74–97%    | 否                   | 详见 arXiv:2511.14617                                     |
| **Relax**    | 小红书              | 全模态（文本+图像+音频）异步训练                                  | 是                   | 详见 arXiv:2604.11554                                     |
| **TRL**      | HuggingFace         | 轻量易用，HuggingFace 生态无缝打通，但不支持大规模异步            | 单轮为主             | [huggingface/trl](https://github.com/huggingface/trl)     |

这些框架的核心差异可以归结为一个取舍：**同步 vs 异步**。同步训练简单、可控、容易调试，但 GPU 利用率低。异步训练吞吐翻倍，但训练数据可能基于旧权重生成，需要额外的算法补偿。

AReaL 的研究表明，异步训练可以在不损失效果的前提下将速度提升近 3 倍——但前提是训练已经调通。Seer 走另一个极端：坚持同步框架，不改变 GRPO 算法，而是通过在线上下文学习（divided rollout、context-aware scheduling、adaptive grouped speculative decoding）消除 rollout 长尾延迟，在保持 on-policy 保证的前提下将吞吐提升 74–97%（[arXiv:2511.14617](https://arxiv.org/abs/2511.14617)）。

另一个关键差异是：框架最初为单轮 RL（推理任务）设计，还是一开始就考虑了多轮 Agent 交互。前者的 Agent 执行模块是后加的，能用但不是为此优化；后者的 Agent 执行是架构一等公民，在状态管理、异构轨迹长度、工具调用异步返回等方面有原生支持。OpenRLHF、AReaL、ROLL、SkyRL 属于后者。

框架选型取决于具体场景。刚入门想快速跑通 demo，OpenRLHF 代码最简洁、文档最完善。企业级大规模训练（70B+），verl 的吞吐和生态优势明显。模型是 MoE 架构（如 GLM-4.5、Qwen3-30B-A3B、DeepSeek-R1），slime 的 Megatron + SGLang 原生架构对 MoE 的 fp8 rollout、DeepEP 通信等做了专门优化。极致吞吐，AReaL 的全异步模式能做到近 3 倍加速。更多工程细节——沙箱管理、环境构建、分布式部署——在 [22.4 工具调用 RL](./tool-use-and-trajectory) 展开。

## 本节总结

Agentic RL 把训练对象从"一段回答"扩展到"一条完整交互轨迹"。这一扩展引出四个核心议题，本章后续小节逐一深入：

- **形式化**——轨迹、状态、动作在多轮设定下如何精确定义？→ [22.2 多轮 RL 形式化](./formulation)
- **信用分配**——一条轨迹最终失败，reward 怎么回拆到每一步？→ [22.3 轨迹信用分配](./credit-assignment)
- **工具与轨迹工程**——训练数据从哪来、工具策略怎么学、沙箱怎么管？→ [22.4 工具调用 RL](./tool-use-and-trajectory)
- **真实训练陷阱**——工业界在哪些坑里摔过？→ [22.6 Code Interpreter RL 工业实战](./industrial-practice)

接下来先看形式化：把"多轮交互"翻译成 RL 能处理的数学对象——[22.2 多轮 RL 形式化](./formulation)。
