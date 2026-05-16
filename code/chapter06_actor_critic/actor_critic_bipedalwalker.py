"""
第6章：用 A2C（Advantage Actor-Critic）训练 BipedalWalker-v3
——从 Pendulum 的 1 维连续动作到 4 维关节协调

运行方式：
    python actor_critic_bipedalwalker.py
    python actor_critic_bipedalwalker.py --total-timesteps 100000    # 快速验证
    python actor_critic_bipedalwalker.py --total-timesteps 3000000   # 充分训练

BipedalWalker-v3 的教学意义：
    1. 24 维状态空间（10 个激光雷达 + 关节角度 + 速度）
    2. 4 维连续动作（两条腿的髋关节和膝关节扭矩）
    3. 比 Pendulum 难得多——需要多关节协调和动态平衡
    4. 展示 Actor-Critic 处理高维连续控制的能力与局限
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


os.makedirs("output", exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="A2C 训练 BipedalWalker-v3")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000,
                        help="总训练步数（默认 3000000）")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="并行环境数量（默认 16）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="最终评估回合数")
    return parser.parse_args()


def make_env(seed, rank):
    def _init():
        env = gym.make("BipedalWalker-v3")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init


class TrainingMonitorCallback(BaseCallback):
    """记录回合奖励、策略熵、价值损失和策略损失。"""

    def __init__(self, checkpoint_steps=None):
        super().__init__()
        self.checkpoint_steps = checkpoint_steps or []
        self.episode_rewards = []
        self.timesteps = []
        self.entropy_losses = []
        self.policy_losses = []
        self.value_losses = []
        self._saved_checkpoints = set()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_info = info["episode"]
                self.episode_rewards.append(
                    ep_info["r"] if isinstance(ep_info, dict) else ep_info
                )

        logger_values = getattr(self.model.logger, "name_to_value", {})
        entropy_loss = logger_values.get("train/entropy_loss")
        policy_loss = logger_values.get("train/policy_loss")
        value_loss = logger_values.get("train/value_loss")

        if entropy_loss is not None and self.num_timesteps not in self.timesteps:
            self.timesteps.append(self.num_timesteps)
            self.entropy_losses.append(float(entropy_loss))
            self.policy_losses.append(float(policy_loss or 0.0))
            self.value_losses.append(float(value_loss or 0.0))

        # 保存检查点
        for ckpt_step in self.checkpoint_steps:
            if self.num_timesteps >= ckpt_step and ckpt_step not in self._saved_checkpoints:
                path = f"output/actor_critic_bipedalwalker_{ckpt_step // 1000}k"
                self.model.save(path)
                print(f"\n  [检查点] 已保存 {ckpt_step // 1000}k 步模型 → {path}.zip")
                self._saved_checkpoints.add(ckpt_step)

        return True


def save_plots(callback, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    rewards = callback.episode_rewards
    if rewards:
        episodes = np.arange(1, len(rewards) + 1)
        window = min(50, max(1, len(rewards)))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x_smooth = np.arange(window // 2, window // 2 + len(smoothed))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, rewards, color="#90CAF9", alpha=0.4, linewidth=0.8,
                label="原始回报")
        ax.plot(x_smooth, smoothed, color="#1565C0", linewidth=1.8,
                label="50 回合滑动平均")
        ax.axhline(y=300, color="green", linestyle="--", alpha=0.5,
                   label="solved (300)")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
        ax.set_title("A2C BipedalWalker-v3 回合奖励", fontsize=14, fontweight="bold")
        ax.set_xlabel("回合")
        ax.set_ylabel("累计奖励")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_bipedalwalker_reward.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  奖励曲线 → output/actor_critic_bipedalwalker_reward.png")

    if callback.entropy_losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(callback.timesteps, callback.entropy_losses,
                color="#EF6C00", linewidth=1.5)
        ax.set_title("A2C BipedalWalker-v3 策略熵损失", fontsize=14, fontweight="bold")
        ax.set_xlabel("时间步")
        ax.set_ylabel("entropy_loss（负熵）")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_bipedalwalker_entropy.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  策略熵曲线 → output/actor_critic_bipedalwalker_entropy.png")

    if callback.policy_losses and callback.value_losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(callback.timesteps, callback.policy_losses,
                color="#00897B", linewidth=1.5, label="Policy loss")
        ax.plot(callback.timesteps, callback.value_losses,
                color="#C62828", linewidth=1.5, label="Value loss")
        ax.set_title("A2C BipedalWalker-v3 Actor/Critic 损失", fontsize=14,
                     fontweight="bold")
        ax.set_xlabel("时间步")
        ax.set_ylabel("损失")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_bipedalwalker_loss.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  损失曲线 → output/actor_critic_bipedalwalker_loss.png")


def main():
    args = parse_args()

    print("=" * 50)
    print("第6章：A2C 训练 BipedalWalker-v3")
    print("=" * 50)
    print(f"总时间步:   {args.total_timesteps:,}")
    print(f"并行环境:   {args.num_envs}")
    print("动作空间:   连续 4 维关节扭矩 [-1, 1]")

    vec_env = DummyVecEnv([make_env(args.seed, i) for i in range(args.num_envs)])

    # A2C 超参数：BipedalWalker 比 Pendulum 更难，需要更多并行环境和更大网络
    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=7e-4,
        n_steps=32,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        verbose=1,
    )

    print(f"\n开始训练（{args.total_timesteps:,} 时间步）...")
    print("-" * 50)

    # 检查点（用于三阶段对比）
    checkpoint_steps = []
    if args.total_timesteps >= 1_000_000:
        checkpoint_steps = [500_000, 1_000_000, 2_000_000]

    callback = TrainingMonitorCallback(checkpoint_steps=checkpoint_steps)
    model.learn(total_timesteps=args.total_timesteps, callback=callback,
                progress_bar=True)

    print("-" * 50)
    print("训练完成！")

    # 保存模型和曲线
    output_dir = Path("output")
    model.save(output_dir / "actor_critic_bipedalwalker")
    print(f"\n模型已保存到 output/actor_critic_bipedalwalker.zip")
    save_plots(callback, output_dir)

    # 评估
    print("\n正在评估最终模型（20 个测试回合）...")
    print("-" * 50)
    eval_env = gym.make("BipedalWalker-v3")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    print(f"20 回合测试结果：")
    print(f"  平均奖励: {mean_reward:.1f}")
    print(f"  标准差:   {std_reward:.1f}")

    test_rewards = []
    for ep in range(args.eval_episodes):
        obs, _ = eval_env.reset(seed=ep)
        total_reward = 0.0
        for step in range(1600):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        test_rewards.append(total_reward)

    print(f"\n逐回合奖励：")
    for i, r in enumerate(test_rewards):
        status = "达标" if r >= 300 else ("中等" if r >= 100 else "未达标")
        print(f"  回合 {i + 1:2d}: {r:8.1f}  [{status}]")

    print(f"\n达标率（>= 300 分）: {sum(1 for r in test_rewards if r >= 300)}/{len(test_rewards)}")
    print(f"  最好一轮: {np.max(test_rewards):.1f}")
    print(f"  最差一轮: {np.min(test_rewards):.1f}")
    eval_env.close()
    print("=" * 50)


if __name__ == "__main__":
    main()
