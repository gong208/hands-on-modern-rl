# B.2 Agentic RL 基础设施 与 沙箱、多轮轨迹与工具调度

> B.1 讨论的是 RL 训练系统底座：rollout、buffer、trainer、权重同步和分布式并行。本页从另一个边界开始：模型的 action 不再只是生成 token，而是会调用工具、运行代码、读写文件、访问网页，或在多轮环境中改变外部状态。

## 与 B.1 的分工

B.2 不重复讲 vLLM/SGLang 如何做高吞吐生成，也不重复讲 FSDP、ZeRO、TP、PP、EP 如何把模型切到多张卡上。这些都是 B.1 的内容。B.2 只讨论 Agentic RL 比普通 LLM RL 多出来的工程问题。

| 问题           | B.1 训练系统底座                                            | B.2 Agentic RL 基础设施                                             |
| -------------- | ----------------------------------------------------------- | ------------------------------------------------------------------- |
| rollout 的含义 | prompt 进入模型，生成 completion 或一段轨迹                 | 模型多轮行动，每步可能调用工具、执行代码或改变环境                  |
| 样本结构       | token、mask、logprob、reward、policy version、rollout batch | episode、tool call、tool result、环境快照、步骤级 reward、loss mask |
| 主要等待       | rollout batch 生成完再训练，或训练后权重回流                | 每一轮工具执行、文件 IO、网络响应、沙箱启动都会让 GPU 空等          |
| 系统重点       | buffer 深度、异步训练、staleness、权重同步、模型并行        | 沙箱隔离、多轮轨迹存储、批内流水线、环境接口、可复现性              |
| 代表框架       | OpenRLHF、veRL、slime                                       | Relax、AReaL、Agent-R1、NeMo Gym                                    |

一句话区分：B.1 解决“训练系统怎么把样本喂给模型”；B.2 解决“Agent 的动作真的发生在外部世界时，训练系统怎么安全、可复现、高吞吐地收集这些动作”。

## 从单轮生成到多轮行动

以第 9 章的 GRPO 训练为例：给定一道数学题，模型一次性生成完整解答，随后检查答案是否正确。整个过程在单轮内完成，模型唯一的操作是文本生成，所有计算均在 GPU 上进行。

相比之下，训练一个具备 bug 修复能力的 Agent 需要完全不同的交互模式。模型拿到一段有错误的代码后，需要先读取代码、定位问题、修改代码、运行测试验证结果。若测试未通过，还需继续修改。一个任务可能需要五六个回合，而每一步之间都存在等待——读文件依赖磁盘 IO，跑测试依赖沙箱执行，搜索依赖网络响应。这些操作不在 GPU 上运行，延迟从几十毫秒到几秒不等。

由此产生了一组环环相扣的工程问题。当 Agent 需要执行代码时，如何确保安全性？这需要**沙箱隔离**。当交互变成多轮之后，如何存储比普通 rollout batch 更复杂的训练数据？这需要**多轮轨迹存储**。当每一轮交互都让 GPU 空闲数百毫秒时，如何避免算力浪费？这需要**GPU 调度优化**。这些问题不是“再多加几个 GPU”就能解决的训练并行问题，而是 Agent 行动本身带来的环境执行问题。下面的讨论将沿着这条链条逐一展开。

## 沙箱隔离

Agent 的核心能力之一是执行代码，这也带来了最大的安全隐患。在训练过程中，模型会尝试各种策略以获取更高分数。如果不加限制，它可能生成 `os.system("rm -rf /")` 来删除训练服务器上的文件，或读取环境变量中的 API key。这些行为并非恶意——模型只是在探索动作空间。但后果是灾难性的：一个运行中的训练任务可能被破坏，导致文件系统被清空、训练数据丢失。

因此，Agent 执行代码必须在*隔离环境*（sandbox）中运行。隔离方案的选择需要在安全性、启动开销和资源利用率之间取得平衡。

### 隔离方案对比

实践中主要有四种隔离方案，适用于不同的场景：

| 方案                   | 隔离粒度               | 启动延迟 | 适用场景                     |
| ---------------------- | ---------------------- | -------- | ---------------------------- |
| subprocess + 资源限制  | 进程级                 | ~10 ms   | 原型验证，信任环境           |
| Docker 容器            | 文件系统 + 网络 + 资源 | ~100 ms  | 通用训练，需要完整隔离       |
| MicroVM（Firecracker） | 内核级                 | ~125 ms  | 安全敏感场景，需要硬件级隔离 |
| WebAssembly（Wasm）    | 指令集级               | ~1 ms    | 纯计算任务，追求极低延迟     |

**subprocess + 资源限制**是最轻量的方案。通过 `rlimit` 限制 CPU 时间和内存、通过 `chroot` 限制文件系统访问、通过 `unshare` 限制网络命名空间。隔离强度有限（进程仍共享宿主内核），但启动开销极低，适合早期原型验证阶段：

```python
import subprocess, resource

def run_in_subprocess(code, timeout=10, max_memory=256 * 1024 * 1024):
    """最轻量的隔离：subprocess + 资源限制"""
    def set_limits():
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))

    result = subprocess.run(
        ["python", "-c", code],
        timeout=timeout,
        preexec_fn=set_limits,
        capture_output=True, text=True,
    )
    return result
```

**Docker 容器**是工业级训练中最常用的方案。通过 Linux cgroups 和 namespace 提供文件系统、网络和资源的三重隔离，且拥有完整的镜像生态（可直接使用 `python:3.11-slim`、`node:20` 等基础镜像）。主要开销在于容器启动（约 100 毫秒），可通过预热容器池优化：

```python
container = client.containers.run(
    "python:3.11-slim",
    command=f"python -c '{code}'",
    detach=True,
    mem_limit="512m",       # 内存限制
    cpu_quota=50000,         # CPU 限制
    network_mode="none",     # 禁止联网
    remove=True,
)
```

**MicroVM（如 Firecracker）** 提供内核级隔离——每个 VM 运行独立的精简 Linux 内核，即使一个 VM 被攻破也无法影响宿主机或其他 VM。AWS Lambda 和 Fly.io 的沙箱即基于此技术。启动延迟约 125 毫秒，与 Docker 相当，但安全性显著更高。适合训练中存在不可信代码执行的场景。

**WebAssembly（Wasm）** 通过 WASI（WebAssembly System Interface）提供一种指令集级别的沙箱。代码被编译为 Wasm 字节码后，只能调用宿主显式导出的函数，无法访问文件系统或网络。启动延迟仅约 1 毫秒，但生态仍在发展中，不支持所有 Python 包。

### 网络策略

无论选择哪种方案，网络访问都需要严格控制。训练中的 Agent 不应直接访问外网——这既不安全，也不可复现（同一个 Agent 第二次运行相同任务时，搜索引擎返回的结果可能已经变化，导致训练轨迹无法重现）。对于确实需要联网的场景（如 Web Agent），应通过代理做请求过滤和缓存，而非简单地断网。

### 预热容器池

当采用 Docker 方案时，容器的启动开销不容忽视。如果同时运行 1000 个 episode，每个都创建新容器，仅启动就需约一分钟。工业级的做法是维护一个*预热容器池*（warm container pool）——提前创建好 N 个容器，用完回收重置而非销毁，启动开销可降至约 5 毫秒。

沙箱解决了"Agent 能安全地执行动作"的问题。接下来，Agent 的多轮交互会产生大量结构化的训练数据，这些数据需要被妥善存储和管理。

## 多轮轨迹存储

### LLM RL vs. Agentic RL 的数据结构差异

B.1 中的 LLM RL 样本并不只是原始文本。真实系统会记录 token ids、attention mask、response mask、old logprob、policy version、reward 等字段。但从结构上看，它仍然接近一条线性序列：`prompt -> completion -> reward`。

Agentic RL 的训练数据则更像一棵带状态的对话树。一个 episode 可能包含七八轮交互，每轮包含模型输出、工具调用参数、工具返回、环境状态变化和步骤级奖励。以"修复 Python bug"任务为例：模型先读代码，然后修改，跑测试发现失败，继续修改，再跑测试通过——这些交互过程都需要完整记录。

### 存储需求

与 B.1 的线性 rollout batch 相比，Agentic RL 的存储系统还需要支持三项额外能力：

- **按任务类型检索**：如分析"数学做得好但代码做得差"的模式
- **按步骤切片**：定位具体哪一步决策出错
- **去重和过期处理**：同一任务不重复训练，旧轨迹可能因环境变化而失效

规模较小时（不到一万条轨迹），JSON 文件加 SQLite 即可满足需求。中等规模（一万到一百万条）可以使用 Redis 做索引、S3 存储数据。超过一百万条则需要分布式数据库（MongoDB 或 DynamoDB）。对于多模态 Agent，轨迹中还包含图片和音频——此时应存储引用（URL）而非原始数据，训练时按需下载，保持轨迹索引在 KB 级别。

存储问题解决后，训练过程中的下一个瓶颈出现了：多轮交互中大量的等待时间导致 GPU 严重空等。

## GPU 空等与异步调度

### 问题量化

B.1 讨论过 LLM RL 的 GPU 空等问题：生成和训练串行执行，训练 GPU 有大量时间在等待生成完成。Agentic RL 将这一问题进一步加剧，而且等待位置发生变化。

以单条 Agentic 轨迹的时间线为例：GPU 生成一个动作约需 3 毫秒，随后 CPU 执行工具约需 500 毫秒。在这 500 毫秒内 GPU 处于空闲状态。下一轮类似：GPU 3 毫秒，CPU 300 毫秒。几轮交互后，GPU 实际工作时间不到 1%。与 LLM RL 相比，LLM RL 的空等发生在 rollout 和 training 之间，而 Agentic RL 的空等发生在每一轮交互内部。

### 批次内并发 与 流水线调度

解法与 B.1 的异步训练思路一脉相承：并发运行多条轨迹。轨迹 A 等待工具返回时，GPU 为轨迹 B 生成动作；轨迹 B 等待工具时，GPU 为轨迹 C 生成动作。通过流水线调度，GPU 持续保持工作状态。这种设计可将 GPU 利用率从约 1% 提升至 70-80%，吞吐量提升 50-100 倍。

### 两级异步

上述方案解决的是"批次内"的并发问题。批次之间仍存在 B.1 讨论的 Rollout 和 Training 串行问题。完整方案采用两级异步：批次内多条轨迹并发（GPU 和工具交替工作），批次间 Rollout 和 Training 通过数据队列解耦（Rollout 持续生成，Training 持续训练）。第一层异步是 Agentic RL 特有的，第二层异步复用 B.1 的训练系统底座。

至此，Agentic RL 的三个基础工程问题——安全执行、数据存储、GPU 调度——已逐一讨论。这些解决方案在真实的工业系统中如何组织？下面的 Relax 案例提供了一个完整的参考实现。

## 工业级实现 与 Relax

Relax 是小红书 AI Infra 团队开源的多模态 Agentic RL 后训练框架，也是目前少数支持全模态（文本、图像、音频）Agentic RL 训练的引擎之一。以下从架构、数据流、执行模式和工程细节四个层面进行分析。

### 分离式架构

Relax 的核心设计选择是将训练流程中的每个角色部署为独立的 Ray Serve 服务：Actor、Rollout、Critic、Reference、Advantages、GenRM 各自独立运行。这一设计源于 Agentic RL 组件的异构性——推理需要 GPU、工具执行需要 CPU、编排需要 CPU 和内存。独立部署使每个组件可以按需扩缩、独立容错，避免资源争用。

```
┌───────────────────────────────────────────────────────────────┐
│  Entrypoints:  train.py                                        │
├───────────────────────────────────────────────────────────────┤
│  Orchestration:  Controller (训练循环) │ Service │ Registry    │
├───────────────────────────────────────────────────────────────┤
│  Components:  Actor │ Rollout │ Critic │ ActorFwd │ GenRM     │
├───────────────────────────────────────────────────────────────┤
│  Engine:  SGLang 推理 │ 奖励函数库 │ 路由 │ 过滤器             │
├───────────────────────────────────────────────────────────────┤
│  Backends:  Megatron-LM (训练) │ SGLang (推理)                 │
├───────────────────────────────────────────────────────────────┤
│  Distributed:  Ray Actor Groups │ DCS (权重同步)               │
└───────────────────────────────────────────────────────────────┘
```

训练后端采用 Megatron-LM，支持 B.1 介绍过的 TP/PP/CP/EP 全套并行策略。推理后端采用 SGLang，两者之间通过 Megatron Bridge 自动完成权重格式转换。

### TransferQueue 与 流式数据通道

回顾 B.1 的异步训练机制：Rollout 生成数据写入 Buffer，Training 从 Buffer 读取数据训练。传统 Buffer 是批量的——Rollout 生成完整个 batch 才写入，Training 等有数据后才读取。这导致一侧始终在等待：Rollout 写入过快时 Buffer 溢出，Training 读取过快时 Buffer 为空导致 GPU 空闲。在 Agentic RL 中，这个问题还叠加了工具执行等待，所以队列最好能承接更细粒度的样本流。

TransferQueue 将这一交互改为流式：Rollout 每生成一个样本即写入队列，Training 端每拿到一个样本即开始处理，无需等待整个 batch 生成完毕。对 Agentic RL 来说，样本不只是 completion，还可能是一条包含多轮工具结果的 episode。配合 DCS（Distributed Checkpoint Service）做权重同步——Training 每更新一步参数，DCS 通过 NCCL 广播给 Rollout 等组件，与下一次训练计算重叠进行，不占额外时间。

这一设计将异步训练中 Batch 级别的等待缩短为 Sample 级别的等待，等待时间降低了一个数量级。

### 两种执行模式

Relax 提供两种模式以适应不同的硬件条件。

*Collocate 模式*下，Actor 和 Rollout 共享同一组 GPU，轮替使用。Rollout 生成完一个 batch，让出 GPU 给 Training。这适合 GPU 数量有限的情况，而且可以做到严格的 on-policy——模型参数没有任何延迟，Training 永远在用最新版本的模型生成的数据。

*Fully Async 模式*下，各角色跑在独立的 GPU 集群上，通过 TransferQueue 交换数据，通过 DCS 异步同步权重。参数 `--max-staleness` 控制允许多"旧"的数据参与训练——设 0 即为严格 on-policy，设大则允许更多异步以换取吞吐。这和 B.1 讨论的"旧数据怎么处理"是同一个底层问题；区别在于 Agentic RL 的"旧"还可能来自环境状态变化、工具版本变化或外部数据变化，因此更需要记录环境快照和可复现信息。

### 工程细节

**Loss mask。** 训练 Agentic RL 时有一个常见的实现误区：将多轮轨迹中的所有 token 都纳入 loss 计算。实际上，工具返回的结果并非模型生成，模型不应为此负责。模型需要学习的是"何时调用何种工具、如何理解工具结果"，而非"如何输出工具结果"。Relax 通过 _loss mask_ 处理这一问题：模型生成的 token 标记为 mask=1 参与训练，工具返回的 token 标记为 mask=0 不参与训练。

**环境接口解耦。** `BaseInteractionEnv` 仅提供 `reset` / `step` / `format_observation` 三个方法，环境实现与 Rollout 逻辑完全分离。更换工具环境无需修改训练代码。虽然这一设计看似理所当然，但在实际项目中，环境与训练逻辑的耦合是非常常见的问题。

**多模态上下文保持。** 多轮对话里，第一轮用户发的图片，到第三轮模型仍需看到。Relax 在 Rollout 端维护 `image_data`，在 Training 端维护 `multimodal_train_inputs`，每轮自动合并。

**弹性扩展。** RL 训练的 60-70% 时间花在 Rollout 上。若训练过程中发现 Rollout 速度成为瓶颈，Relax 支持在不中断训练的情况下动态增加推理引擎：

```bash
# 在当前集群里加引擎
curl -X POST http://controller:8000/scale \
  -d '{"target_engine_count": 4, "mode": "ray_native"}'

# 或者注册其他集群已有的引擎（跨集群联邦推理）
curl -X POST http://controller:8000/scale \
  -d '{"engine_urls": ["gpu-cluster-2:8000"], "mode": "external"}'
```

`external` 模式值得注意——它可以利用其他 GPU 集群上的空闲资源或抢占式实例来加速 Rollout，无需将它们迁移到当前集群。

### 算法、模型和运维

**算法支持。** Relax 内置了四种算法：GRPO（见 [8.1-8.2 节](/chapter18_grpo/grpo-practice-and-mechanism)）、GSPO、SAPO 和 OPD（见 [8.5 节](/chapter18_grpo/on-policy-distillation)）。添加新算法只需实现一个 Service 类并注册到 `ALGOS` 字典。

**模型支持。** Qwen3 全系列（4B、30B-A3B MoE）、Qwen3-VL（视觉语言）、Qwen3-Omni（全模态）和 Qwen3.5。

**运维体系。** HealthManager 负责心跳监控和两级自动恢复（先尝试原地重启，失败后全局重启）；Metrics Service 将训练指标分发到 TensorBoard / WandB / ClearML；Apprise 负责推送告警到 Slack、微信、邮件。大规模 RL 训练的挑战不在于启动，而在于持续稳定运行——一个训练任务可能持续数天甚至数周，期间 GPU 故障、网络抖动、OOM 都是常态。缺乏自动恢复机制将导致运维人员频繁手动介入，严重影响训练效率。

### 与其他框架对比

| 框架     | 出品方             | 特点                              | 多模态 | 异步       |
| -------- | ------------------ | --------------------------------- | ------ | ---------- |
| AReaL    | Ant Group 和清华   | 全异步，2.77x 提速                | 否     | 全异步     |
| Seer     | Moonshot AI (Kimi) | 极致同步，rollout 吞吐 +74–97%    | 否     | 同步       |
| Agent-R1 | 中科大             | MDP 扩展，过程/结果奖励分离       | 否     | 部分异步   |
| NeMo Gym | NVIDIA             | 科学 Agent 环境                   | 否     | 同步为主   |
| slime    | THUDM / 智谱生态   | Megatron + SGLang，MoE 原生优化   | 否     | 支持异步   |
| Relax    | 小红书             | TransferQueue + 弹性扩展 + 全模态 | 是     | 全异步流式 |

Relax 是目前唯一同时支持全模态和全异步弹性扩展的 Agentic RL 引擎。Seer 则代表了另一个方向——不走向异步，而是在同步框架内通过在线上下文学习（divided rollout、context-aware scheduling、adaptive grouped speculative decoding）消除 rollout 长尾延迟，在不改变 GRPO 算法的前提下将吞吐提升 74–97%，同时保持严格的 on-policy 保证（[arXiv:2511.14617](https://arxiv.org/abs/2511.14617)）。slime 把 SGLang 作为原生推理层、Megatron 作为训练后端，对 GLM-4.5、Qwen3-30B-A3B、DeepSeek-R1 等 MoE 模型做了 fp8 rollout、DeepEP 通信等专项优化，适合 MoE 架构的大规模后训练（[THUDM/slime](https://github.com/THUDM/slime)）。Relax 论文见 [arxiv.org/abs/2604.11554](https://arxiv.org/abs/2604.11554)，代码见 [github.com/redai-infra/Relax](https://github.com/redai-infra/Relax)。

## 选型建议

以下是实践层面的选型建议。原型验证阶段，TRL 配合 subprocess 即可满足需求——先验证训练流程的可行性和 reward 信号的正确性。需要开箱即用的全流程支持（SFT → DPO/GRPO → 部署）时，[ms-swift](https://github.com/modelscope/ms-swift) 提供了 ModelScope 生态的一体化方案，适合快速上手和国产模型适配。中等规模（如数百条轨迹并发）可以采用 veRL 或 OpenRLHF，配合 Docker 沙箱和 asyncio 实现异步并发。大规模 Agentic 训练则需要 Relax 或 AReaL 等全异步框架。多模态 Agent 场景下，Relax 是目前唯一的选择。

建议遵循渐进式架构演进原则：先验证流程可行性，再做性能优化，最后进行生产化改造。

## nanoRLHF — 从零实现一个 LLM RL 训练框架

前面的框架分析都是站在使用者视角：每个组件做什么，数据怎么流，GPU 怎么调度。但要真正理解一个 RL 训练框架的内部结构，最好的办法是自己写一个。[hyunwoongko/nanoRLHF](https://github.com/hyunwoongko/nanoRLHF)（181 stars）正是这样一个项目——它用纯 PyTorch + Triton 从零实现了 LLM RLHF 训练所需的全部组件，包括训练引擎、推理引擎、分布式调度和 RL 编排。

nanoRLHF 的定位类似 nanoGPT：把一个生产系统剥离到只保留承重结构。它的目录结构直接对应了 B.1 讨论的系统层级：

```
nanorlhf/
├── nanotron/     # 训练引擎（3D parallelism、gradient accumulation、checkpoint）
├── nanovllm/     # 推理引擎（PagedAttention、KV cache、continuous batching）
├── nanoverl/     # RL 编排层（PPO trainer、reward、dataset、configs）
├── nanoray/      # 分布式调度（进程管理、资源分配）
├── nanosets/     # 数据集工具
├── kernels/      # Triton kernel（fusion、优化算子）
└── eval/         # 评测工具
```

### nanotron

nanotron 对应 B.1 中的"训练/编排层"的底层——它负责把大模型拆到多张 GPU 上训练。核心实现了 3D parallelism（数据并行 + 流水并行 + 张量并行）、gradient accumulation、mixed precision training 和 checkpoint 管理。

阅读入口：`nanotron/` 目录。重点关注：

- 张量并行如何把一个线性层拆到多张卡上（`nanotron/parallel`）
- 流水并行如何把模型的不同层分到不同设备（`nanotron/pipeline`）
- 梯度累积和梯度同步在分布式场景下如何协调

### nanovllm

nanovllm 对应 B.1 中的"推理/rollout 层"——它负责高吞吐生成 token。核心实现了 PagedAttention（vLLM 的关键技术）、KV cache 管理和 continuous batching。

阅读入口：`nanovllm/` 目录。重点关注：

- PagedAttention 如何避免 KV cache 的显存浪费
- continuous batching 如何让不同长度的请求共享 GPU
- 这里的推理引擎和训练引擎之间的权重如何对接

### RL 编排 与 nanoverl

nanoverl 是把前面两个引擎串起来的编排层，对应 B.1 中 OpenRLHF/veRL 的角色。它实现了 PPO 训练循环：rollout（用 nanovllm 生成）→ reward 计算 → advantage 估计 → PPO clipped loss → 梯度更新（用 nanotron 训练）。

阅读入口：`nanoverl/trainer/` 目录。重点关注：

- PPO 的 fit() 循环如何编排 actor、reference、rollout 三个角色
- KL 散度惩罚如何实现（reference model 作为锚点）
- reward 函数如何接入（数学验证场景）

### 推荐阅读路线

整个项目可以按以下顺序阅读，从底层到上层：

1. **`nanotron/`** — 先理解训练引擎如何做分布式训练，因为这是框架的地基
2. **`nanovllm/`** — 再看推理引擎如何做高吞吐生成，理解 rollout 端的工程问题
3. **`nanoverl/`** — 最后看 RL 编排如何把两者串成 PPO 循环，理解"生产者-消费者"的数据流
4. **`nanoray/`** — 如果对分布式调度感兴趣，看进程管理和资源分配

### 动手练习

```bash
# 克隆项目
git clone https://github.com/hyunwoongko/nanoRLHF.git
cd nanoRLHF

# 安装依赖（需要 CUDA GPU）
pip install -e .
```

建议完成以下练习：

1. **跑通 SFT 训练**：`bash ./scripts/train_sft.sh`，观察训练日志中的 loss、lr、throughput 指标
2. **阅读 PPO trainer**：打开 `nanoverl/trainer/`，画出 rollout → reward → advantage → train 的数据流图
3. **对比 B.1 的框架表**：把 nanoRLHF 的每个模块对应到 OpenRLHF / veRL / slime 的等价组件，理解抽象边界的异同
4. **修改 reward 函数**：在 `nanoverl/reward/` 中替换为自己的 reward 逻辑（例如字符串匹配、正则提取），跑通一个自定义 reward 的 RL 训练循环

nanoRLHF 的价值不在于生产使用，而在于它用可读的代码把 B.1 讨论的"rollout engine、training backend、weight sync、policy version"这些概念变成了具体实现。读完之后再看 veRL 或 OpenRLHF 的源码，会快得多。

## 参考文献

[^relax_paper]: Zhang L, Ning B, Yang R, et al. "[Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](https://arxiv.org/abs/2604.11554)." arXiv:2604.11554, 2026. [GitHub](https://github.com/redai-infra/Relax)

[^1]: HuggingFace Blog, "[Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape)", 2026.

[^2]: PyTorch Blog, "[A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/)", 2025.

[^3]: AReaL Team. "[AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)." arXiv:2505.24298, 2025. [GitHub](https://github.com/inclusionAI/AReaL)

[^4]: Hou L et al. "[Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning](https://arxiv.org/abs/2511.14617)." arXiv:2511.14617, 2025.

[^5]: Ko H. "[nanoRLHF: From-scratch journey into how LLMs and RLHF really work](https://github.com/hyunwoongko/nanoRLHF)." GitHub, 2025.
