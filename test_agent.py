"""
Test and compare trained agents.

Usage:
  # Test a single checkpoint
  python test_agent.py --checkpoint ray_results/PPO_SP_vs_Baseline/PPO_Soccer_xxx/checkpoint_000100

  # Compare two checkpoints head-to-head
  python test_agent.py \
    --checkpoint ray_results/PPO_SP_vs_Baseline/PPO_Soccer_xxx/checkpoint_000100 \
    --compare    ray_results/PPO_test/PPO_Soccer_yyy/checkpoint_000100

  # Test vs baseline opponent instead of stationary
  python test_agent.py --checkpoint ... --opponent baseline
"""
import argparse
import os
import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import soccer_twos
from soccer_twos import EnvType

from baseline_policy import load_baseline_policy


def load_policy(checkpoint_path):
    """Load a trained PPO policy from a Ray 2.x checkpoint."""
    config = (
        PPOConfig()
        .environment(
            env="Soccer",
            disable_env_checking=True,
            env_config={
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
                "base_port": 50700,
            },
        )
        .framework("torch")
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=0)
        .training(
            model={"vf_share_layers": True, "fcnet_hiddens": [512, 512, 256]},
        )
    )
    from ray.tune.registry import register_env
    from utils import create_rllib_env
    register_env("Soccer", create_rllib_env)

    algo = config.build()
    algo.restore(checkpoint_path)
    policy = algo.get_policy()

    def act(obs):
        action, _, _ = policy.compute_single_action(obs)
        return action

    return act


def run_episodes(policy_fn, opponent_fn, num_episodes=10, base_port=50700, label="agent"):
    """Run episodes and return stats."""
    env = soccer_twos.make(
        variation=EnvType.team_vs_policy,
        single_player=True,
        flatten_branched=True,
        opponent_policy=opponent_fn,
        base_port=base_port,
        worker_id=0,
    )

    rewards = []
    wins = 0
    losses = 0
    draws = 0

    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0.0

        while not done:
            action = policy_fn(np.array(obs, dtype=np.float32))
            result = env.step(action)
            obs, reward, done = result[0], result[1], result[2]
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward += reward

        rewards.append(total_reward)
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

        print(f"  [{label}] Episode {ep+1}/{num_episodes} — reward: {total_reward:+.3f}")

    env.close()

    print(f"\n  {label} over {num_episodes} episodes:")
    print(f"    Mean reward : {np.mean(rewards):+.4f}")
    print(f"    Std         : {np.std(rewards):.4f}")
    print(f"    W/D/L       : {wins}/{draws}/{losses}")
    return {"mean": np.mean(rewards), "std": np.std(rewards), "wins": wins, "draws": draws, "losses": losses}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint folder to test")
    parser.add_argument("--compare", type=str, default=None,
                        help="Optional second checkpoint to compare against")
    parser.add_argument("--opponent", type=str, default="stationary",
                        choices=["stationary", "baseline"],
                        help="Opponent to test against (default: stationary)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per agent")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)

    # Pick opponent
    if args.opponent == "baseline":
        print("Loading CEIA baseline as opponent...")
        opponent_fn = load_baseline_policy()
        opponent_label = "vs baseline"
    else:
        opponent_fn = lambda *_: 0
        opponent_label = "vs stationary"

    # Infer a readable name from the checkpoint path
    def infer_name(path):
        parts = path.rstrip("/").split("/")
        for p in parts:
            if "PPO_SP_vs_Baseline" in p:
                return "trained-vs-baseline"
            if "PPO_test" in p or "PPO_SP" in p:
                return "trained-vs-stationary"
        return os.path.basename(path)

    print(f"\n{'='*55}")
    print(f"Testing: {args.checkpoint}")
    print(f"Opponent: {opponent_label}")
    print(f"{'='*55}")

    policy_a = load_policy(args.checkpoint)
    name_a = infer_name(args.checkpoint)
    stats_a = run_episodes(policy_a, opponent_fn, args.episodes, base_port=50700, label=name_a)

    if args.compare:
        print(f"\n{'='*55}")
        print(f"Comparing: {args.compare}")
        print(f"{'='*55}")
        policy_b = load_policy(args.compare)
        name_b = infer_name(args.compare)
        stats_b = run_episodes(policy_b, opponent_fn, args.episodes, base_port=50800, label=name_b)

        print(f"\n{'='*55}")
        print(f"COMPARISON SUMMARY ({opponent_label})")
        print(f"{'='*55}")
        print(f"  {name_a:30s}  mean={stats_a['mean']:+.4f}  W/D/L={stats_a['wins']}/{stats_a['draws']}/{stats_a['losses']}")
        print(f"  {name_b:30s}  mean={stats_b['mean']:+.4f}  W/D/L={stats_b['wins']}/{stats_b['draws']}/{stats_b['losses']}")
        winner = name_a if stats_a["mean"] > stats_b["mean"] else name_b
        print(f"\n  Better agent: {winner}")

    ray.shutdown()
