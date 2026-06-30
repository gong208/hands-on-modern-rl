---
title: 22. Agentic RL
---

# Chapter 22: Agentic RL

The RL problems covered in earlier chapters are essentially **single-turn decision-making**: the model receives a prompt, emits a complete response, a reward model scores it, and the policy is updated once. Whether the underlying algorithm is PPO or GRPO, the skeleton of "one prompt, one response, one score" never changes.

Real agents do not work this way.

Consider a flight-booking agent. The user says "Book me the cheapest early-morning flight from Beijing to Shanghai tomorrow." The agent must act in steps: search for flights, compare prices and times, confirm seat availability, call the order API, and wait for ticket confirmation. Any step gone wrong—an overly broad search query, skipping price comparison, misjudging inventory, malformed order parameters—fails the whole task. The environment gives a single binary signal at the end: ticket issued (reward = 1) or not (reward = 0).

This shift from "one-shot QA" to "multi-step interaction with an environment" is the core problem Agentic RL addresses.

## The Paradigm Shift from Single-Turn to Multi-Turn

The flight-booking example reveals four new challenges absent from single-turn RL:

1. **The training object changes from completion to trajectory.** A trajectory mixes model-generated tokens, tool calls, tool returns, and environment state changes—structurally closer to a dialogue tree than a linear piece of text.
2. **Rollouts must execute in a real environment.** Each step can trigger external calls (search, API, code execution); the GPU has to wait for the environment, with utilization as low as 20–30%.
3. **Environments must be modular, resettable, and verifiable.** You cannot actually book ten thousand tickets during training—you need sandboxes or simulators.
4. **Multi-turn training is more prone to instability.** A 10-step trajectory has reward only at the final step; earlier good and bad decisions are painted with the same reward brush—this is the **credit assignment problem**.

This chapter unfolds around these four challenges. We first build intuition with two contrastive trajectories.

## Two Contrastive Trajectories

Same flight-booking task, same model, two rollouts:

```
Trajectory A (success)                   Trajectory B (failure)
─────────────────────────────            ─────────────────────────────
T1 search("Beijing Shanghai              T1 search("Beijing Shanghai flight")
      early morning cheap flight")          obs: 200 mixed results
   obs: 12 relevant flights

T2 filter(dep<9:00, sort=price)          T2 pick_first()
   obs: CA1501 6:30 ¥760                    obs: MU5101 9:30 ¥1280

T3 check_seat(CA1501)                    T3 order(MU5101)
   obs: seats available                     obs: order placed

T4 order(CA1501, seat=window)
   obs: ticket issued

reward = 1                               reward = 0
```

The two trajectories end with very different rewards, but **which step is to blame**? Did Trajectory B fail because T1's query was too broad, because T2 picked the first option without comparing prices, or because T3 ordered without confirmation? The final reward alone cannot answer. We will return to these two trajectories throughout the chapter, examining them from different angles (component decomposition, MDP formalization, credit assignment).

## Basic Components of an Agent

An agent is more than an LLM. Minimal definition: **LLM backbone + instructions + tools + environment**, circulating in an agentic loop.

### LLM Backbone

The decision-making core of the agent. It receives the current observation, reasons about the next step, and produces an action (text or tool call). Any sufficiently strong LLM can serve as a backbone, but in practice we often choose reasoning-trained models—they emit a thinking trace before producing an action, which is friendlier to multi-step decisions.

### Instructions

Tell the agent what problem to solve and what strategy to use. Beyond the task itself ("book the cheapest early-morning flight"), this includes problem-solving hints ("search first, then filter," "balance price against time," "retry on failure"). The quality of instructions directly determines the floor of agent behavior.

### Tools and Environment

Tools are the agent's interface to the environment: search APIs, code interpreters, CLIs, MCP servers, order APIs. Tool calls are typically delimited by special tokens embedded in the model's token stream:

```
<tool_call>{"name":"search_flights","args":{"from":"PEK","to":"SHA"}}</tool_call>
<tool_response>[CA1501 6:30 ¥760, CA1831 7:00 ¥690, ...]</tool_response>
```

The environment is stateful: search results change, inventory shifts, placing an order mutates a database. A tool call's return depends not only on its arguments but also on the environment's current state. This ability to anchor outputs to the real world rather than parametric memory is called **grounding**—a major advantage of agents over pure LLMs, and a core behavioral pattern that RL training can instill.

### Agentic Loop

The four components cycle: **observe → reason about the next step → execute an action → receive a new observation**, until a termination condition is met (task complete, max steps reached, or the model emits an end signal).

A complete loop is called a **rollout**; the full interaction record produced by a rollout is called a **trajectory**, denoted $\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T)$. A trajectory is not a text sequence—it mixes model-generated tokens, tool calls, tool returns, and environment state changes, structurally closer to a dialogue tree than linear text.

## MDP Formalization of Agentic RL

The previous section described agents conceptually. To train one, we need to write this interaction as an RL problem. The most natural approach is to start from the familiar single-turn MDP and extend step by step.

### The Single-Turn RL MDP

The GRPO algorithm from earlier chapters is essentially a **degenerate MDP**:

- **State** $s$: the current token context (prompt + generated tokens so far)
- **Action** $a$: the next token
- **Transition** $P$: deterministic append—the chosen token is added to the context
- **Reward** $r$: given once after the entire rollout finishes (usually by a reward model or verifier)
- **Trajectory** $\tau$: the complete token sequence

The optimization objective is $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$—make the single response as good as possible.

### The Multi-Turn RL MDP

Extend each of the single-turn components:

- **State** expands: token context **plus** external environment state $s_t = (c, x_{1:t}, e_t)$, where $c$ is the task instruction, $x_{1:t}$ is the token history, and $e_t$ is the current environment state (e.g., a flight database snapshot). This is a **joint state**.
- **Action** expands: text tokens **plus** structured tool calls $A = A_{\text{text}} \cup A_{\text{action}}$.
- **Transition** expands: no longer deterministic—the environment may be stochastic (search results change), tools may fail, and sampling itself introduces noise.
- **Reward** expands: can be given only at the end (ORM) or at every step (PRM).

This "model sees only part of the state" setting corresponds to a **partially observable Markov decision process (POMDP)**:

$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle
$$

where $O$ is the observation function, $o_t = O(s_t)$—the model sees observations, not the full state.

|                | Single-Turn RL (GRPO)                  | Multi-Turn Agentic RL                                            |
| -------------- | -------------------------------------- | ---------------------------------------------------------------- |
| **State**      | A single prompt; episode ends at once  | Joint state $(c, x_{1:t}, e_t)$, evolving with interaction       |
| **Action**     | Plain text tokens                      | Text + structured tool calls                                     |
| **Transition** | Deterministic append                   | Dynamic; environment may be non-deterministic                    |
| **Reward**     | One scalar $r(a)$                      | Step-level or terminal; often sparse task-completion signal      |
| **Objective**  | $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$ | $\mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t R(s_t, a_t)]$ |

The formalization itself is not complicated. The key point: **the innovation in Agentic RL often lies not in "the RL formula itself" but in "the system design that lets RL act on a real agent loop"**—how to define state and action, how to design rewards, how to handle long-horizon credit assignment.

## Credit Assignment and Step-Level Signals

Back to the two flight-booking trajectories. Trajectory B failed with reward = 0. But T2's "pick the first option without comparing prices" is clearly the main error—how do we turn this intuition into a training signal?

**Credit assignment** breaks the final trajectory reward back down into a step-level advantage for each step.

### Three-Layer Signal Decomposition

From final outcome to token update, the signal passes through three layers:

| Layer             | Meaning                                      | Form                                      |
| ----------------- | -------------------------------------------- | ----------------------------------------- |
| Trajectory reward | The final $R(\tau)$ for the whole trajectory | Only tells you "did this episode succeed" |
| Step advantage    | Decomposed into per-step $A_t$               | Answers "should step t be rewarded"       |
| Token gradient    | $A_t$ multiplied into this action's log-prob | Actually updates the LLM weights          |

Naive trajectory-level RL uses $R(\tau)$ to update all tokens, effectively saying "all actions in a successful trajectory are good, all in a failed one are bad." Credit assignment corrects this coarse attribution.

### ORM and PRM

The simplest scheme is **ORM (Outcome Reward Model)**—reward only at the end. The advantage is clear signal and cheap annotation: a verifier automatically checks "does the answer match" or "do the tests pass," with no need to label every step. RLVR (Reinforcement Learning with Verifiable Rewards) is the extreme form of ORM: skip training a reward model entirely and use a binary verifier. The success of DeepSeek-R1 demonstrates that pure RLVR can elicit strong reasoning ability.

But ORM cannot distinguish the responsibility of T2 vs T3 in Trajectory B—the whole trajectory scores 0.

**PRM (Process Reward Model)** scores each step independently. OpenAI's "Let's Verify Step by Step" (Lightman et al., 2023) formalized this idea: step 1 is correct (+1), step 2 has a calculation error (−0.5), step 3 reaches the right answer but takes a detour (+0.3). The model can precisely locate which steps need improvement. The cost is annotation: OpenAI built the PRM800K dataset for this purpose. Current research focuses on **automated PRM** (e.g., Math-Shepherd), where models judge step quality on their own.

In practice, ORM and PRM are often **combined**: ORM supplies a reliable final-outcome signal; PRM supplies dense mid-process guidance.

### Relative Advantage of Multiple Actions at the Same State

Finer-grained credit assignment doesn't just look at absolute scores—it **compares different actions at the same state**. Consider the T2 decision point in Trajectory B: under the same observation of 200 flights, three possible actions:

| Action                            | Subsequent outcome        | $R_t$ |
| --------------------------------- | ------------------------- | ----- |
| Compare prices, pick the cheapest | Subsequent success        | 1.00  |
| Skip comparison, pick the first   | Subsequent failure        | 0.00  |
| Page through and re-search        | Eventually succeeds, slow | 0.65  |

The group mean is 0.55, giving relative advantages:

```
Compare prices, pick the cheapest:  1.00 - 0.55 = +0.45
Skip comparison, pick the first:    0.00 - 0.55 = -0.55
Page through and re-search:         0.65 - 0.55 = +0.10
```

What the model learns is not "all steps in a successful trajectory are good," but "at this state, comparing prices is best; skipping comparison is worst; paging is so-so." This **state-anchored group** idea underlies a wave of 2026 methods including GiGPO and HGPO—see [Multi-Turn RL and Credit Assignment](./multi-turn-rl) for details.

## Training Mechanics

Regardless of algorithmic details, RL training alternates between two operations:

1. **Rollout**: sample a batch of trajectories using the current policy $\pi_\theta$ and compute each reward.
2. **Policy update**: use PPO / GRPO / REINFORCE to compute advantages from the rollouts and update $\theta$.

Agentic rollouts are far more complex than single-turn ones:

- **Long horizon**: a flight-booking rollout may take 5–10 steps; a code agent may exceed 20.
- **Heterogeneous length**: the same prompt can yield rollouts of very different lengths—simple tasks finish in 3 steps, complex ones in 15+.
- **Environment latency**: each tool call waits for the environment, leaving the GPU idle much of the time.

These engineering issues—asynchronous training, sandbox management, heterogeneous trajectory batching, long-tail latency elimination—are what Agentic RL frameworks in 2025–2026 aim to solve. The next section expands on this.

::: tip Beyond Credit Assignment: A Training Pitfall
In addition to step-level signals, Agentic RL faces **reward hacking**—the model exploits loopholes in the reward function instead of actually solving the problem. For example, if a code agent's reward only checks "do tests pass," the model may learn to generate a mock function that always returns `True`. Engineering countermeasures are covered in [Industrial Practice](./industrial-evaluation).
:::

## A Minimal Agent Loop

Reading concepts ten times is worse than running them once. Let's build a runnable agent in a few dozen lines—no RL training, just "how does an agent interact with tools?" Once this loop is clear, adding RL follows naturally.

```python
import json, subprocess, os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

# ① Define tools: tell the model "what you can do"
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

# ② Tool execution logic (the environment)
def execute_tool(name, args):
    if name == "execute_bash":
        r = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
        return r.stdout + r.stderr
    elif name == "read_file":
        with open(args["path"]) as f:
            return f.read()
    return f"Unknown tool: {name}"

# ③ Agent Loop: perceive → reason → act → observe, repeat
def run_agent(task, max_turns=5):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": task},
    ]
    for turn in range(max_turns):
        # Perceive + reason: model decides next step given current info
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)

        # Act: if no tool call, the model produced a final answer
        if not msg.tool_calls:
            return msg.content  # Agent considers the task done; exit the loop

        # Observe: execute the tool and feed the result back
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  [Turn {turn+1}] Tool call: {tc.function.name}({args})")
            result = execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "(Max turns reached; stopping.)"

# ④ Try it
print(run_agent("List the .md files in the current directory and tell me how many there are."))
```

Expected output:

```
  [Turn 1] Tool call: execute_bash({'command': 'ls *.md'})
  [Turn 2] Tool call: execute_bash({'command': 'ls *.md | wc -l'})
There are 12 .md files in the current directory.
```

Map this 50-line code to the earlier concepts:

- `tools = [...]` corresponds to the **action space** $A_{\text{action}}$—the set of tools the agent can call, the new action type added by Agentic RL on top of single-turn RL.
- `execute_tool()` corresponds to the **environment**—the actual execution logic. The agent says "execute bash," the environment returns command output.
- `for turn in range(max_turns)` corresponds to the **Agentic Loop / Rollout**—each iteration is one step $(s_t, a_t, o_{t+1})$; the entire `for` loop is one complete trajectory sample.
- `client.chat.completions.create()` corresponds to the **policy** $\pi_\theta$—the model decides what to do next: which tool to call, what arguments to pass. Currently fixed weights; after RL training, this gets optimized.
- `messages.append(...)` corresponds to the **state** $s_t$—the full dialogue history is the current state; the model sees all prior interactions.

How "smart" this agent is depends entirely on the policy $\pi_\theta$. The current model is a pretrained general model; it knows "when to use a bash command" because it saw many examples during pretraining and SFT. But it does not know: for this specific task, how to construct the most efficient search query? When the first search returns nothing useful, should it reformulate the query or change strategy? These "strategic decisions" are exactly what RL optimizes.

How to add RL training to this agent—how to compute rewards (ORM vs PRM), how to distribute reward across multi-step interaction (credit assignment), how to manage training data—is the subject of subsequent sections. In [Multi-Turn RL and Credit Assignment](./multi-turn-rl), we extend this simple loop into a trainable RL system.

## The Limits of SFT and Prompting

A natural question: ReAct, Toolformer, and similar methods already let LLMs call tools. Why do we still need RL?

The key distinction: SFT and prompting teach the model to **imitate**—copying patterns of "when to call tools, which tool to call" from human demonstrations. But in real agent tasks, the optimal strategy for tool use is highly context-dependent:

- How to construct search queries? When to open page details? When to stop searching and start summarizing?
- After a code edit, if tests still fail, should we keep debugging or switch direction?
- When multiple sources contradict each other, which one to trust?

These are fundamentally **strategy-learning problems**, not pure language modeling. Demonstration data cannot cover every possible decision path, while RL can shape tool-use, planning, and memory-management behaviors from task outcomes.

The division of labor between SFT and RL in agentic scenarios:

- **SFT teaches format**: the syntax of tool calls (e.g., the JSON format for calling a search engine), the basic interaction protocol.
- **RL teaches strategy**: when to call a tool, how to compose multi-step actions, how to recover from failure.

The DeepSeek-R1-Zero experiment suggests that skipping SFT and going straight to RL can also surface reasoning ability—provided the base model is strong enough. In practice, the two-stage recipe of SFT warmup + RL fine-tuning remains the dominant paradigm.

## Industrial Framework Landscape

Back to reality—when you actually want to train an agent, which framework do you use?

For PPO and GRPO in earlier chapters, this question was not sharp: the training loop is almost entirely GPU compute, and either TRL or OpenRLHF handles it easily. But the Agentic RL training loop adds a "wait"—when the model calls a search engine, the GPU waits for results; when the model runs code, the GPU waits for the sandbox. How to keep the GPU from idling? This is the core problem Agentic RL frameworks solve.

In 2025–2026, a batch of open-source frameworks emerged around this problem:

| Framework    | Developer             | One-line description                                                                                               | Native multi-turn agent support | GitHub                                                    |
| ------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------- |
| **OpenRLHF** | Open-source community | Most concise code (~8k lines); algorithm decoupled from agent execution; one line to switch single-turn/multi-turn | Yes                             | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |
| **verl**     | ByteDance / community | Highest throughput; training and inference dynamically share the same GPU group; richest ecosystem                 | Basic, community-extending      | [verl-project/verl](https://github.com/verl-project/verl) |
| **slime**    | THUDM / Z.ai ecosystem | Megatron + SGLang post-training framework for RL scaling; strong MoE efficiency                                    | Basic                           | [THUDM/slime](https://github.com/THUDM/slime)             |
| **AReaL**    | Ant Group / Tsinghua  | Fully asynchronous training—GPU never waits; 2.77× speedup                                                         | Yes                             | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) |
| **ROLL**     | Alibaba Taotian       | RLVR + Agent dual mode; native Qwen support                                                                        | Yes                             | [alibaba/ROLL](https://github.com/alibaba/ROLL)           |
| **SkyRL**    | UC Berkeley           | Modular full-stack—training, agent orchestration, task environments independently                                  | Yes                             | [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)   |
| **Seer**     | Moonshot AI (Kimi)    | Pushed synchronous—eliminates rollout long-tail via in-context learning; 74–97% throughput gain                    | No                              | see arXiv:2511.14617                                      |
| **Relax**    | Xiaohongshu           | Fully multimodal (text + image + audio) asynchronous training                                                      | Yes                             | see arXiv:2604.11554                                      |
| **TRL**      | HuggingFace           | Lightweight and easy to use; seamless HF ecosystem integration; no large-scale async support                       | Mostly single-turn              | [huggingface/trl](https://github.com/huggingface/trl)     |

The core difference between these frameworks comes down to a single trade-off: **synchronous vs asynchronous**. Synchronous training is simple, controllable, and easy to debug, but GPU utilization is low. Asynchronous training doubles throughput, but training data may be generated from stale weights, requiring extra algorithmic compensation. AReaL's research shows async training can deliver nearly 3× speedup without quality loss—provided training is already healthy. Seer takes the opposite extreme: it sticks with a synchronous framework, leaves GRPO untouched, and instead eliminates rollout long-tail latency via in-context learning (divided rollout, context-aware scheduling, adaptive grouped speculative decoding), achieving 74–97% throughput gains while preserving on-policy guarantees ([arXiv:2511.14617](https://arxiv.org/abs/2511.14617)).

Another key difference: was the framework originally designed for single-turn RL (reasoning tasks), or did it account for multi-turn agent interaction from the start? The former's agent execution module was added later—usable but not optimized for it; the latter treats agent execution as a first-class architectural citizen, with native support for state management, heterogeneous trajectory lengths, and asynchronous tool-call returns. OpenRLHF, AReaL, ROLL, and SkyRL fall into the latter camp.

Framework choice depends on the scenario. For beginners wanting to get a demo running quickly, OpenRLHF has the most concise code and best docs. For enterprise-scale training (70B+), verl's throughput and ecosystem win. For MoE models (GLM-4.5, Qwen3-30B-A3B, DeepSeek-R1), slime's Megatron + SGLang native architecture optimizes fp8 rollout and DeepEP communication for MoE. For maximum throughput, AReaL's fully async mode delivers ~3× speedup. More engineering details—sandbox management, environment construction, distributed deployment—are covered in [Tool Use and Agentic Engineering](./tool-use-and-trajectory).

## Chapter Structure

::: tip Prerequisites
This chapter uses the following concepts frequently; review them first:

- [GRPO and RLVR](../chapter18_grpo/rlvr)—"verifiable rewards" are a natural choice for Agentic RL
- [PPO and Reward Models](../chapter10_ppo/intro)—the foundational policy optimization framework
  :::

| Section                                                                                  | Core question                                                                                                   |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [Multi-Turn RL and Credit Assignment](./multi-turn-rl)                                   | After 7 turns the task failed—whom do we blame? ORM vs PRM; planning ability; hands-on labs                     |
| [Tool Calling, Trajectory Synthesis, and Agentic Engineering](./tool-use-and-trajectory) | Where does training data come from? When should the model use tools? Sandbox, async rollout, reward design      |
| [Industrial Practice, Evaluation, and Bad Cases](./industrial-evaluation)                | How does real training go unstable? How to localize problems via benchmarks, eval pipelines, and bad-case loops |
| [Agent Data Fabrication—SWE-smith](./agent-data-swe-smith)                               | Auto-generate 50k+ code agent training data: inject bugs, run tests, filter useful samples                      |
| [Hands-on Lab: Training a DeepCoder Agent with rLLM](./rllm-deepcoder-lab)               | rLLM in practice: AgentFlow + sandbox verification + GRPO RL training                                           |
| [Project 2: Deep Research Agent](./projects)                                             | Long-horizon search, citation verification, report generation, and Deep Research RL schemes                     |
| [Hands-on: Building an Agentic Training System](./build-agentic-training-system)         | Build Environment + Policy + RolloutWorker + Trainer from scratch; understand the framework skeleton            |
| [Further Reading Index](./extended-readings)                                             | 13 topic clusters, 120+ papers—an open index for going deeper                                                   |

---

Next, let's tackle the most central problem in multi-step interaction: when the final outcome fails, which step should bear the blame?—[Multi-Turn RL and Credit Assignment](./multi-turn-rl).
