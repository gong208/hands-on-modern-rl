# 第 36 章 · 分布式 RL 训练系统

> [附录 B.1 RL 训练系统](../appendix_industrial_training/rl-infrastructure) 已经讲了基础——采样、异步、分布式并行。本章把视角提升到**框架级架构**和**前沿工业实践**：veRL 如何用 HybridFlow 统一编排多模型、AReaL/LlamaRL 如何用纯异步打破生成-训练壁垒、DeepSeek V3 的 DualPipe 如何在 MoE 上做流水线并行、万卡集群如何 profile 与调优。

## 36.1 veRL 架构深度解析

veRL（Volcano Engine Reinforcement Learning）是字节跳动 2024 年开源的 RL 训练框架，论文 [HybridFlow, arXiv:2409.19256](https://arxiv.org/abs/2409.19256)。它已经成为事实上的主流 LLM RL 训练框架，被 Qwen、DeepSeek、Llama、Mistral 等团队的训练脚本采用。

### HybridFlow 的核心设计

HybridFlow 把 RLHF/GRPO/PPO 训练抽象成 **single-controller 多模型编排**：

```
┌─────────────────────────────────────────────────────────┐
│              Single Controller (Driver)                  │
│  - 算法逻辑（PPO/GRPO 算法主循环）                       │
│  - 资源调度（哪些 GPU 跑哪个模型）                       │
└──────────┬──────────────────────────────────────────────┘
           │
   ┌───────┼───────┬─────────────┬─────────────┐
   │       │       │             │             │
   ▼       ▼       ▼             ▼             ▼
┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Actor │ │Critic│ │Reference │ │Reward    │ │Rollout   │
│(FSDP)│ │(FSDP)│ │(Frozen)  │ │Model     │ │Engine    │
│      │ │      │ │          │ │          │ │(vLLM)    │
└──────┘ └──────┘ └──────────┘ └──────────┘ └──────────┘
   ▲       ▲       ▲             ▲             ▲
   │       │       │             │             │
   └───────┴───────┴─────────────┴─────────────┘
              ResourcePool (GPU 集合)
```

### 三个核心抽象

#### 1. ResourcePool

把 GPU 分组，每组可以放一个或多个模型：

```python
# veRL 配置示例（简化）
resource_pools = {
    "actor_pool": num_gpus=8,    # Actor 用 8 张卡
    "critic_pool": num_gpus=4,   # Critic 用 4 张卡
    "rollout_pool": num_gpus=8,  # Rollout 用 8 张卡
    "ref_pool": num_gpus=2,      # Reference 模型 2 张卡
}
```

不同模型可以**共享 GPU**（colocate）或**独占 GPU**（disaggregated）：

```python
# Colocate：Actor 和 Rollout 共享同一组 GPU
mapping = {
    "actor": "actor_rollout_pool",
    "rollout": "actor_rollout_pool",  # 共享！
    "critic": "critic_pool",
    "ref": "ref_pool",
}
```

#### 2. Worker

每个 Worker 是一个独立的模型实例，封装了具体的训练/推理逻辑：

```python
class ActorWorker:
    def __init__(self, model_config):
        self.model = FSDPActor(model_config)

    def update(self, batch):
        # PPO/GRPO loss 计算 + 反向传播
        loss = compute_ppo_loss(batch, self.model)
        loss.backward()
        self.optimizer.step()

    def get_weights(self):
        # 给 Rollout Engine 同步权重
        return self.model.state_dict()

class RolloutWorker:
    def __init__(self, model_config):
        self.engine = vLLMEngine(model_config)

    def generate(self, prompts):
        return self.engine.generate(prompts)

    def sync_weights(self, new_weights):
        self.engine.load_weights(new_weights)
```

#### 3. Driver（Single Controller）

Driver 是 RL 算法的主循环，编排所有 Worker：

```python
class PPODriver:
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # 1. 让 Actor 暴露当前权重给 Rollout
            weights = self.actor_worker.get_weights()
            self.rollout_worker.sync_weights(weights)

            # 2. 用当前策略采样
            prompts = sample_prompts(self.dataset)
            responses = self.rollout_worker.generate(prompts)

            # 3. 用 Reward Model 算 reward
            rewards = self.reward_worker.score(prompts, responses)

            # 4. 用 Critic 算 value
            values = self.critic_worker.value(prompts, responses)

            # 5. 算 advantage + PPO loss 更新 Actor
            advantages = compute_gae(rewards, values)
            self.actor_worker.update(prompts, responses, advantages)

            # 6. 更新 Critic
            self.critic_worker.update(prompts, responses, rewards)
```

### HybridFlow 的"Hybrid"含义

Hybrid 指**统一的混合并行策略**——同一个框架内可以组合：

- **3D Parallelism**：TP（张量并行）× PP（流水线并行）× DP（数据并行）
- **Colocate vs Disaggregated**：模型可共享或独占 GPU
- **多种训练后端**：FSDP、Megatron、DeepSpeed ZeRO
- **多种推理后端**：vLLM、SGLang、HuggingFace generate

veRL 是第一个把这些维度都做成可配置的框架。DeepSpeed-Chat、OpenRLHF 在某些维度上更受限。

### 与其他框架的核心差异

| 维度 | veRL (HybridFlow) | OpenRLHF | NeMo-Aligner | TRL |
|------|-------------------|----------|--------------|-----|
| **编排方式** | Single-controller | Single-controller | Multi-controller | Single-process |
| **资源分配** | 任意组合 | 严格分离 | NVIDIA 栈 | 单 GPU |
| **训练后端** | FSDP + Megatron | FSDP/DeepSpeed | Megatron | Accelerate |
| **推理后端** | vLLM/SGLang | vLLM | TRT-LLM | HF generate |
| **典型规模** | 8-1024 GPU | 8-256 GPU | 8-512 GPU | 1-8 GPU |

[第 9 章 GRPO 实践](../chapter09_grpo_rlvr/grpo-practice-and-mechanism) 用的就是 veRL。

## 36.2 OpenRLHF / NeMo-Aligner / TRL 对比

### OpenRLHF

[OpenRLHF, arXiv:2405.11143](https://arxiv.org/abs/2405.11143) 由 OpenLLMAI 团队维护，是最早的开源 RLHF 框架之一。

**核心设计**：

- 基于 **Ray** 做分布式调度
- 严格的 **Actor/Critic/Ref/RM 分离**——每个模型在独立的 Ray Actor 进程
- 强调**简洁性**和**易用性**

```python
# OpenRLHF 训练 PPO（伪代码）
from openrlhf import PPOTrainer, ModelGroup

actor = ModelGroup(num_gpus=8, backend="deepspeed")
critic = ModelGroup(num_gpus=8, backend="deepspeed")
ref = ModelGroup(num_gpus=4)
reward = ModelGroup(num_gpus=4)
vllm = VLLMRollout(num_gpus=8)

trainer = PPOTrainer(actor, critic, ref, reward, vllm)
trainer.train(dataset, num_epochs=100)
```

**适用场景**：研究用途、中等规模训练（8-256 GPU）。SimpleRL、Llama-3.1 后训练都用过 OpenRLHF。

### NeMo-Aligner

[NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) 是 NVIDIA 官方栈，深度集成 Megatron-LM 和 TRT-LLM。

**核心设计**：

- **Megatron** 训练后端（最强的大模型并行能力）
- **TRT-LLM** 推理后端（NVIDIA 自家的推理优化）
- 偏好 NVIDIA 全栈优化

**适用场景**：NVIDIA 集群、超大模型（70B+）、追求极致性能。Nemotron 系列、Llama-3 在 NVIDIA 集群上的训练都用 NeMo。

### TRL (Transformer Reinforcement Learning)

[TRL](https://github.com/huggingface/trl) 是 HuggingFace 出品的轻量级框架。

**核心设计**：

- 基于 **Accelerate**（HuggingFace 的分布式抽象）
- 单进程模型，靠 Accelerate 自动切分
- **易用性第一**：10 行代码跑 PPO

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
config = PPOConfig(batch_size=8)
trainer = PPOTrainer(config, model)
trainer.train(dataset)
```

**适用场景**：学习、原型验证、小规模实验（1-8 GPU）。不适合生产级训练。

### 四框架对比

| 框架 | 易用性 | 性能 | 规模上限 | 工业采用 |
|------|--------|------|---------|---------|
| **veRL** | 中 | 高 | 1024+ GPU | Qwen、DeepSeek、字节内部 |
| **OpenRLHF** | 高 | 中 | 256 GPU | SimpleRL、部分开源 |
| **NeMo-Aligner** | 低 | 极高 | 512+ GPU | NVIDIA 客户、Nemotron |
| **TRL** | 极高 | 低 | 8 GPU | 研究、教学 |

**推荐选择**：

- 学习、原型：TRL
- 研究、中等规模：OpenRLHF 或 veRL
- 大规模生产：veRL 或 NeMo-Aligner（看硬件栈）

## 36.3 Rollout 引擎与 vLLM 集成

RL 训练 99% 的时间在 rollout（[附录 B.1](../appendix_industrial_training/async-training)）。Rollout 引擎是性能瓶颈的核心。vLLM 是事实标准。

### vLLM 的核心优化

#### 1. PagedAttention

传统 KV cache 是连续分配，导致显存碎片严重。vLLM 借鉴 OS 的分页机制，把 KV cache 分成固定大小的 block：

```python
# 传统：KV cache 连续分配
seq_len = 2048
kv_cache = torch.empty(batch_size, seq_len, num_heads, head_dim)
# 显存利用率 50-70%

# vLLM PagedAttention：分块
block_size = 16
blocks = allocate_blocks(num_blocks)
# 显存利用率 95%+
```

显存利用率从 50-70% 提升到 95%+，batch size 提升 2-4 倍。

#### 2. Continuous Batching

传统 batching 是"等一个 batch 全部生成完才换"。vLLM 是**动态 batching**——某条序列生成完后立刻换上新序列：

```
时间:  ──────────────────────────────────────►
序列A: [tok][tok][tok][tok][EOS]
序列B: [tok][tok][tok][tok][tok][tok][EOS]
序列C:           [tok][tok][tok][tok][EOS]  ← A 结束后立刻加入
序列D:                    [tok][tok][tok][EOS]  ← C 结束后加入
```

吞吐提升 5-10 倍 vs 静态 batching。

#### 3. Speculative Decoding

用小模型先 draft 几个 token，大模型并行验证：

```python
def speculative_decode(prompt, draft_model, target_model, num_draft=4):
    while not done:
        # 1. 小模型生成 num_draft 个 token
        draft_tokens = draft_model.generate(prompt, max_tokens=num_draft)

        # 2. 大模型并行验证
        target_logits = target_model.forward(prompt + draft_tokens)

        # 3. 接受匹配的 token，拒绝后重新生成
        for i, token in enumerate(draft_tokens):
            if target_logits[i].argmax() == token:
                prompt.append(token)
            else:
                prompt.append(target_logits[i].argmax())
                break
```

吞吐提升 2-3 倍（典型 LLM 推理）。

### vLLM 在 RL 训练中的角色

veRL 中 vLLM 作为 RolloutWorker：

```python
class VLLMRolloutWorker:
    def __init__(self, model_path, tensor_parallel_size=8):
        from vllm import LLM
        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,  # 关键：GRPO 同 prompt 多采样时复用 KV
            gpu_memory_utilization=0.9,
        )

    def generate(self, prompts, sampling_params):
        # 批量生成
        return self.engine.generate(prompts, sampling_params)

    def sync_weights(self, new_weights):
        # vLLM 0.5+ 支持在线权重更新
        self.engine.load_weights(new_weights)
```

**Prefix Caching** 对 GRPO 特别重要——同一个 prompt 生成 $G=8$ 条回答，前缀（prompt 部分）的 KV cache 可以复用，节省 70-80% 的显存和时间。

### SGLang：vLLM 的挑战者

[SGLang](https://github.com/sgl-project/sglang) 由 LMSYS 团队开发，在 agentic 场景下比 vLLM 更快：

- **RadixAttention**：用基数树管理 KV cache，跨请求复用
- **Programmatic Frontend**：支持复杂的控制流（多轮调用、分支、循环）
- **Constrained Decoding**：内置 JSON、regex 约束生成

工业实践中：

- **vLLM**：通用 rollout、单轮生成
- **SGLang**：agentic rollout、多轮、结构化输出
- **TRT-LLM**：NVIDIA 硬件极致优化

## 36.4 GPU 内存优化：ZeRO、FSDP、Gradient Checkpointing

LLM 训练显存是核心瓶颈。一个 70B 模型 bf16 全参训练需要 ~1.5 TB 显存——单卡 80GB H100 根本放不下。

### 显存分解

训练显存包含四部分：

$$\text{Memory} = \underbrace{|\theta| \cdot 2}_{\text{权重（bf16）}} + \underbrace{|\theta| \cdot 2}_{\text{梯度}} + \underbrace{|\theta| \cdot 8 + \text{optimizer state}}_{\text{Adam state}} + \underbrace{\text{activation}}_{\text{激活}}$$

对 70B 模型：

- 权重：140 GB
- 梯度：140 GB
- Adam state（m, v, master weights）：560 GB
- 激活：~100 GB（取决于 batch size 和 seq len）
- **总计**：~940 GB

单卡 80GB H100 远远不够。

### ZeRO (Zero Redundancy Optimizer)

[DeepSpeed ZeRO, arXiv:1910.02054](https://arxiv.org/abs/1910.02054) 把训练状态切分到多个 GPU：

| 阶段 | 切分内容 | 节省倍数 | 通信开销 |
|------|---------|---------|---------|
| **ZeRO-1** | Optimizer state | 4× | 低 |
| **ZeRO-2** | Optimizer + Gradient | 8× | 中 |
| **ZeRO-3** | Optimizer + Gradient + Weight | $N$×（N=GPU 数） | 高 |

ZeRO-3 把权重也切分，每个 GPU 只存 $1/N$ 的权重，但前向反向时需要 all-gather 还原。

```python
# DeepSpeed ZeRO-3 配置
config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
    },
    "bf16": {"enabled": True}
}
```

### FSDP (Fully Sharded Data Parallel)

PyTorch 原生的 ZeRO-3 等价物，比 DeepSpeed 更易用：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = LlamaForCausalLM(config)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 等价 ZeRO-3
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    cpu_offload=CPUOffload(offload_params=False),  # 可选 CPU offload
)
```

veRL 默认用 FSDP——比 DeepSpeed 更稳定、与 PyTorch 生态更兼容。

### Gradient Checkpointing

不切分模型，而是用计算换显存——前向时不保存中间激活，反向时重新计算：

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        # 用 checkpoint 包裹 transformer block
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.transformer_block(x)
```

激活显存从 $O(L)$ 降到 $O(\sqrt{L})$（$L$ 是层数），代价是前向计算两次——训练慢 20-30%。

### 三者组合的内存账

对 70B 模型（8 张 H100 80GB）：

| 配置 | 单卡显存 | 训练速度 |
|------|---------|---------|
| 全参 + Adam（baseline） | 940 GB（炸） | - |
| ZeRO-3 | 118 GB（炸） | - |
| ZeRO-3 + Gradient Checkpointing | 30 GB | 1× |
| ZeRO-3 + Gradient Checkpointing + LoRA | 8 GB | 1.2× |

LoRA（[第 8 章](../chapter08_rlhf/industrial-post-training)）只训少量参数，显存需求大幅降低。工业级 70B RL 训练通常用 LoRA + FSDP。

## 36.5 异步 RL 训练

同步训练的瓶颈在 [附录 B.1](../appendix_industrial_training/async-training) 详述——GPU 99% 空闲等 rollout。异步训练把生成和训练解耦，让两边同时跑。下面介绍三个 2025 年的旗舰框架。

### LlamaRL

[LlamaRL, Meta arXiv:2506.10910](https://arxiv.org/abs/2506.10910) Meta 2025 年 6 月发布的分布式 RL 框架：

**核心创新**：**完全去中心化**——没有 master node，每个 worker 自治。

```python
# LlamaRL 架构（简化）
class LlamaRLWorker:
    def run(self):
        while True:
            # 每个 worker 自己决定做什么
            if self.role == "rollout":
                prompts = self.fetch_from_queue()
                responses = self.generate(prompts)
                self.push_to_train_queue(responses)

            elif self.role == "train":
                batch = self.fetch_from_rollout_queue()
                self.update(batch)
                self.broadcast_weights()  # 异步广播
```

**优势**：

- 无单点故障
- 横向扩展容易（加 worker 即可）
- 适合超大规模（10k+ GPU）

**实测**：在 4096 GPU 上跑 Llama-3-70B GRPO，比同步训练快 **10.4×**。

### AReaL (Asynchronous RL)

[AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning, arXiv:2505.24298](https://arxiv.org/abs/2505.24298) 是 Ant Group 和清华 2025 年开源的大规模异步 LLM RL 系统：

**核心创新**：**fully asynchronous rollout + staleness-aware PPO**。Rollout worker 持续生成样本，training worker 在拿到 batch 后立即消费；系统通过控制样本 staleness，并在 PPO 更新中加入针对旧策略样本的校正，缓解“生成策略已经落后当前训练策略 K 步”的偏移。

```python
# AReaL 关键算法（简化）
def staleness_aware_update(batch, current_weights):
    # batch 记录了 rollout 时的 policy version 与 logprob
    gen_log_probs = batch["gen_log_probs"]
    current_log_probs = compute_log_probs(batch, current_weights)
    importance_weights = torch.exp(current_log_probs - gen_log_probs)

    # 截断重要性权重，避免旧样本造成过大梯度
    clipped_weights = torch.clamp(importance_weights, 0.8, 1.2)
    loss = -(clipped_weights * advantages).mean()

    return loss
```

**优势**：

- 允许训练用旧数据，不要求严格 on-policy
- 缓冲区可以积累大量数据
- 训练和生成完全解耦

**实测**：在 1024 GPU 上跑 671B MoE GRPO，比同步快 **2.77×**。

### AgentRL

[AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework, arXiv:2510.04206](https://arxiv.org/abs/2510.04206) 是 2025 年 10 月发布的多轮、多任务 Agentic RL 框架，代码在 [THUDM/AgentRL](https://github.com/THUDM/AgentRL)：

**核心创新**：**fully-asynchronous generation-training pipeline + 统一环境接口**。训练侧用 rollout、actor、reference 三类 worker 池异步采样和更新；环境侧用 function-call API、containerized environment、centralized controller 和 task worker 管理异构任务。算法侧还引入 cross-policy sampling 和 task advantage normalization，分别增强多轮探索并稳定多任务训练。

```python
# AgentRL 异步训练结构（简化）
rollout_workers.stream_trajectories(task_manager)
actor_workers.update_policy(buffer.sample())
reference_workers.compute_kl(buffer.sample())
controller.route_function_calls(task_workers)
```

**优势**：

- 支持 multi-turn、multi-task agentic RL
- 异步解耦 trajectory 采集和 policy 更新
- 通过 controller / task worker / transport layer 管理环境部署
- 被用于构建 AutoGLM

**适用场景**：SWE-Agent、Computer Use、Deep Research Agent 训练。

### 三大异步框架对比

| 框架 | 主要贡献者 | 核心机制 | 加速比 | 适用场景 |
|------|-----------|---------|--------|---------|
| **LlamaRL** | Meta | 完全去中心化 | 10.4× | 超大规模 Dense |
| **AReaL** | Ant Group 和清华 | 全异步 rollout + staleness-aware PPO | 2.77× | 大规模 LLM RL |
| **AgentRL** | THUDM/智谱 | 多轮多任务 + 统一环境接口 | 论文未标注 | Agent 训练 |

## 36.6 MoE + RL 训练

DeepSeek V3、Qwen3、GLM-4.5 都是 MoE 架构。MoE 给 RL 训练带来新挑战。

### MoE 的特殊性

MoE 模型的参数分布不均匀——大多数参数在 expert 里，每条样本只激活少数 expert：

```
MoE 模型结构（DeepSeek V3）:
┌─────────────────────────────────────┐
│ Dense 部分（attention 等）: 20B 参数 │
├─────────────────────────────────────┤
│ MoE 部分:                            │
│  - 256 个 expert × 5B 参数 = 1.28T   │
│  - 每条样本激活 8 个 expert           │
│  - 实际激活参数: 40B                  │
└─────────────────────────────────────┘
总参数: 1.3T，激活参数: 60B
```

### MoE RL 训练的三个挑战

#### 1. Expert 负载不均

某些 expert 被频繁激活，其他 expert 闲置。导致：

- 计算负载不均（部分 GPU 过载）
- 训练数据分布偏（部分 expert 训练不充分）

**解决**：**Expert Balancing Loss**：

```python
def expert_balancing_loss(router_logits, num_experts):
    # 计算每个 expert 的激活频率
    router_probs = torch.softmax(router_logits, dim=-1)
    expert_freq = router_probs.mean(dim=0)  # [num_experts]

    # 鼓励均匀分布
    target_freq = 1.0 / num_experts
    balance_loss = ((expert_freq - target_freq) ** 2).mean()

    return balance_loss
```

#### 2. 通信开销

MoE 的 expert 分布在多个 GPU（Expert Parallelism），每条样本都要 all-to-all 通信：

```
GPU 0: expert 0,1,2     ──┐
GPU 1: expert 3,4,5     ──┼── all-to-all ── 处理完后 all-to-all 回去
GPU 2: expert 6,7,8     ──┤
GPU 3: expert 9,10,11   ──┘
```

**解决**：**DeepEP**（DeepSeek Expert Parallelism），优化 all-to-all 通信模式。

#### 3. Token 级 IS 方差大

[GRPO 家族](../chapter09_grpo_rlvr/grpo-family) 提到——MoE 下不同 token 路由到不同 expert，token 级 importance sampling 比率波动剧烈，梯度方差大。

**解决**：**GSPO（Group Sequence Policy Optimization）**——把 IS 比率从 token 级改成序列级：

```python
# PPO/GRPO: token 级 IS
token_ratio = exp(log_prob_new - log_prob_old)  # 每个 token 独立

# GSPO: 序列级 IS
sequence_log_prob_new = sum(log_prob_new_per_token)
sequence_log_prob_old = sum(log_prob_old_per_token)
sequence_ratio = exp(sequence_log_prob_new - sequence_log_prob_old)
# 整个序列用同一个 ratio
```

Qwen3 全系（包括 235B-A22B）都基于 GSPO 训练。

### DeepSeek V3 的 MoE RL

DeepSeek V3（671B MoE，37B 激活）的 RL 训练实践：

- **DualPipe**：流水线并行优化（详见 36.7）
- **FP8 训练**：用 FP8 减少显存和计算（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）
- **MTP (Multi-Token Prediction)**：一次预测多个 token，提升训练信号密度

### Step Flash（阶跃星辰）

Step Flash 是阶跃星辰 2025 年发布的 MoE RL 优化：

- **Dynamic Expert Allocation**：根据 batch 内 token 分布动态调整 expert 数量
- **Sparse Gradient Sync**：只同步被激活的 expert 的梯度
- **Cache-aware Routing**：路由时考虑 KV cache 局部性

### GLM-4.5（智谱）

GLM-4.5 用 **slime** 框架训练（[THUDM/slime](https://github.com/THUDM/slime)）：

- Megatron 训练后端
- SGLang 推理后端
- 原生 MoE 优化（DeepEP 通信、fp8 rollout）

## 36.7 DualPipe 与 Best-Fit Packing

### DualPipe

[DeepSeek V3 论文 arXiv:2412.19437](https://arxiv.org/abs/2412.19437) 提出 **DualPipe**——双向流水线并行。

传统流水线并行（PP）的气泡（bubble）问题：

```
GPU 0: [F0][F1][F2][F3]              [B3][B2][B1][B0]
GPU 1:       [F0][F1][F2][F3]   [B3][B2][B1][B0]
GPU 2:             [F0][F1][F2][F3][B3][B2][B1][B0]
                   ↑                ↑
                   前向              反向
                   气泡很大
```

DualPipe 让前向和反向**同时跑**——前向 stage N 和反向 stage N-1 在同一 GPU 上重叠：

```
GPU 0: [F0|B0][F1|B1][F2|B2][F3|B3]  ← 前向和反向重叠
GPU 1:       [F0|B0][F1|B1][F2|B2][F3|B3]
GPU 2:             [F0|B0][F1|B1][F2|B2][F3|B3]
                                    几乎没有气泡
```

气泡比例从传统的 $\frac{P-1}{M}$（$P$ 是 PP stage 数，$M$ 是 micro-batch 数）降到 $\frac{P-1}{2M}$。

```python
# DualPipe 伪代码
class DualPipeScheduler:
    def schedule(self, num_stages, num_micro_batches):
        schedule = []
        for step in range(num_micro_batches + num_stages - 1):
            for stage in range(num_stages):
                # 同一 stage 同一 step 既做前向又做反向
                fwd_mb = step - stage
                bwd_mb = step - (num_stages - 1 - stage)
                if fwd_mb >= 0 and fwd_mb < num_micro_batches:
                    schedule.append(("forward", stage, fwd_mb))
                if bwd_mb >= 0 and bwd_mb < num_micro_batches:
                    schedule.append(("backward", stage, bwd_mb))
        return schedule
```

### Best-Fit Packing

传统 micro-batch 分配是均匀的——每个 GPU 拿相同数量。但 MoE 下不同 expert 负载不同，均匀分配导致不均衡。

**Best-Fit Packing**：用装箱算法（bin packing）把不同大小的 micro-batch 分配到 GPU：

```python
def best_fit_pack(items, bin_capacity):
    """items 是不同大小的 micro-batch, bin_capacity 是单 GPU 容量"""
    bins = [[]]
    for item in sorted(items, reverse=True):  # 从大到小
        # 找到能放下且最满的 bin
        best_bin = None
        best_remaining = float('inf')
        for bin in bins:
            remaining = bin_capacity - sum(bin)
            if item <= remaining < best_remaining:
                best_bin = bin
                best_remaining = remaining
        if best_bin is None:
            bins.append([item])
        else:
            best_bin.append(item)
    return bins
```

DeepSeek V3 用 Best-Fit Packing 让 GPU 利用率从 70% 提升到 95%。

## 36.8 性能 Profiling 与瓶颈分析

RL 训练的性能优化必须基于 profiling——不能凭感觉。

### Profiling 工具

#### 1. PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    trainer.train_step()

# 打印 top 10 耗时操作
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

#### 2. NVIDIA Nsight Systems

```bash
# 用 nsys 跑训练
nsys profile -o rl_train_profile python train.py

# 用 Nsight Systems GUI 查看时间线
nsys-ui rl_train_profile.qdrep
```

可视化每个 CUDA kernel 的执行时间、CPU-GPU 同步、通信开销。

#### 3. veRL 内置 Profiler

veRL 提供了 RL 特定的 profiling：

```python
from verl.utils.profiler import RLProfiler

with RLProfiler() as p:
    trainer.train()
    p.print_summary()
# 输出：
#   rollout time: 3500s (85%)
#   actor update time: 120s (3%)
#   critic update time: 80s (2%)
#   weight sync time: 30s (0.7%)
#   communication: 400s (10%)
```

### 典型瓶颈与优化

| 瓶颈 | 症状 | 优化 |
|------|------|------|
| **Rollout 慢** | rollout 占 80%+ 时间 | 增加 rollout GPU、用 vLLM prefix caching |
| **Weight Sync 慢** | sync 占 5%+ 时间 | 用 LoRA、NCCL 打包传输 |
| **通信开销** | all-reduce 占 10%+ 时间 | 增大 batch size、用 gradient accumulation |
| **激活显存爆炸** | OOM | Gradient checkpointing |
| **Expert 负载不均** | 部分 GPU 90%+、部分 30% | Expert balancing loss、动态路由 |
| **慢人问题** | batch 内最长序列决定时间 | 长度分桶、Seer divided rollout |

### MFU (Model FLOPs Utilization)

衡量训练效率的金标准：

$$\text{MFU} = \frac{\text{实际 FLOPs}}{\text{峰值 FLOPs} \times \text{时间}}$$

H100 bf16 峰值 ~1000 TFLOPS。典型 LLM RL 训练 MFU：

| 配置 | MFU |
|------|-----|
| Dense + FSDP + checkpointing | 35-45% |
| MoE + EP + DualPipe | 50-60% |
| 异步 RL（生成/训练分离） | 70-80%（rollout 部分用 vLLM 加速） |

MFU < 30% 说明有显著优化空间——通常是通信或 rollout 瓶颈。

## 36.9 万卡集群实践

把上面所有技术组合起来——这就是 2025 年万卡集群上的 RL 训练实践。

### 典型配置

以 Qwen3-235B-A22B（235B 总参，22B 激活 MoE）的 GRPO 训练为例：

```yaml
# 集群配置
total_gpus: 12288  # 12k H100
intra_node_bandwidth: 900 GB/s  # NVLink
inter_node_bandwidth: 50 GB/s   # InfiniBand

# 模型并行
tensor_parallel: 8        # TP=8（节点内）
pipeline_parallel: 4      # PP=4（跨节点）
expert_parallel: 16       # EP=16
data_parallel: 24         # DP=24

# 训练配置
algorithm: GSPO            # MoE 优化的 GRPO 变体
batch_size_per_gpu: 1
gradient_accumulation: 32
seq_len: 32768
group_size: 8              # GRPO 每个 prompt 生成 8 条

# 异步配置
async_mode: disaggregated
rollout_buffer_size: 100000
weight_sync: lora          # 只同步 LoRA adapter
weight_sync_method: nccl_packed
```

### 实测性能

```text
训练 1 epoch (10B tokens):
  Total time: 24 小时
  GPU hours: 294912
  
分项时间:
  Rollout: 18 小时 (75%)
  Actor update: 3 小时 (12.5%)
  Critic update: 2 小时 (8%)
  Weight sync: 0.5 小时 (2%)
  Other: 0.5 小时 (2.5%)

MFU: 52%（MoE + DualPipe + FP8）
```

### 万卡训练的关键经验

#### 1. 故障是常态

12288 张卡，平均每天有 5-10 张故障。必须：

- **Checkpoint 频率**：每 30 分钟存一次，故障时回滚
- **冗余设计**：每 1024 张卡配 8 张备份
- **自动重启**：故障检测后自动从最近 checkpoint 恢复

#### 2. 通信是性能杀手

跨节点通信慢，万卡集群网络设计：

- **Topology-aware**：相邻 GPU 优先组成 tensor parallel group
- **Overlap 通信与计算**：反向传播时同时启动梯度 all-reduce
- **Gradient Bucket**：合并小梯度，减少通信次数

#### 3. MoE 的路由稳定性

MoE 训练中 expert 路由可能突然塌缩——所有 token 都路由到少数 expert。监控：

```python
# 实时监控 expert 负载
def monitor_expert_balance(model):
    while training:
        for layer in model.moe_layers:
            router_probs = layer.router.get_recent_probs()
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10))
            if entropy < threshold:  # 路由熵过低
                alert(f"Layer {layer.id}: expert routing collapse!")
        time.sleep(60)
```

#### 4. 数据 pipeline 是隐藏瓶颈

万卡集群每秒消费数百万 token，数据加载本身可能成为瓶颈：

- **预取**：提前准备未来 10 个 batch 的数据
- **数据压缩**：用更紧凑的格式存储
- **分布式存储**：数据分布在多个 SSD，避免单点 I/O 瓶颈

## 本章总结

分布式 RL 训练系统是 LLM 时代的核心工程：

1. **veRL (HybridFlow)** 是主流框架——single-controller 多模型编排，灵活的资源分配
2. **OpenRLHF/NeMo-Aligner/TRL** 各有定位——研究、NVIDIA 栈、轻量教学
3. **vLLM/SGLang** 是 rollout 引擎的核心——PagedAttention、Continuous Batching、Prefix Caching
4. **ZeRO/FSDP/Checkpointing** 解决显存——LoRA + FSDP + Checkpointing 是 70B 训练标配
5. **异步训练（LlamaRL/AReaL/AgentRL）** 是 2025 年的方向——10× 加速、Off-policy 容忍
6. **MoE + RL** 需要 GSPO、Expert Balancing、DualPipe、Best-Fit Packing 协同优化
7. **万卡集群** 是工程极限——故障常态、通信瓶颈、监控告警、数据 pipeline

[第 17 章 LLM RL 工业实战](../chapter09_alignment/industrial-post-training) 会从产品视角再讲一遍这些技术如何落地——这一章是工程视角。

## 延伸阅读

- [Sheng et al. 2024 "HybridFlow: A Flexible and Efficient RLHF Framework"](https://arxiv.org/abs/2409.19256)
- [Hu et al. 2024 "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework"](https://arxiv.org/abs/2405.11143)
- [Kwon et al. 2023 "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)](https://arxiv.org/abs/2309.06180)
- [Rajaseharan et al. 2024 "SGLang"](https://arxiv.org/abs/2312.07104)
- [Rajbhandari et al. 2020 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054)
- [Zhao et al. 2025 "LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework"](https://arxiv.org/abs/2506.10910)
- [Fu et al. 2025 "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"](https://arxiv.org/abs/2505.24298)
- [Zhang et al. 2025 "AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework"](https://arxiv.org/abs/2510.04206)
- [DeepSeek-AI 2024 "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437)
- [DeepSeek-AI 2025 "DeepSeek-R1: Incentivizing Reasoning Capability via RL"](https://arxiv.org/abs/2501.12948)
- [Qwen Team 2025 "Qwen3 Technical Report"](https://arxiv.org/abs/2505.09388)
- [Zhao et al. 2024 "GSPO: Group Sequence Policy Optimization"](https://arxiv.org/abs/2507.18071)
- [Seer Team 2025 "Seer: Eliminating Rollout Tail Latency"](https://arxiv.org/abs/2511.14617)
