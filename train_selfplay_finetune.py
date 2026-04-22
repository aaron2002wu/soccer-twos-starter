"""
Phase 3: Self-play fine-tuning from a pre-trained checkpoint.
Restores from the best checkpoint of phase 1 or 2, then switches to
multiagent_team self-play to develop coordination and passing.

Usage:
  python train_selfplay_finetune.py --checkpoint path/to/checkpoint
"""
import os
import io
import logging
import warnings
import argparse
import yaml

os.environ["RAY_DASHBOARD_ENABLED"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env, sample_pos_vel, sample_player


NUM_WORKERS         = 20
NUM_ENVS_PER_WORKER = 1
TRAIN_BATCH         = 10000
TARGET_STEPS        = 15_000_000

W_BALL_DIST = 0.02
W_BALL_GOAL = 0.03
W_SHOOT     = 0.05
W_BLOCK     = 0.002
W_SPREAD    = 0.002


class SelfPlayCallback(DefaultCallbacks):

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        agents = episode.get_agents()
        obs_map  = {a: episode.last_observation_for(a) for a in agents}
        prev_map = {a: episode._agent_to_last_obs.get(a) for a in agents}

        for agent_id in agents:
            obs      = obs_map[agent_id]
            prev_obs = prev_map[agent_id]
            if prev_obs is None:
                continue

            o = obs[:336]  if len(obs)      >= 336 else obs
            p = prev_obs[:336] if len(prev_obs) >= 336 else prev_obs

            ball_x,      ball_y      = float(o[7]), float(o[8])
            prev_ball_x, prev_ball_y = float(p[7]), float(p[8])
            ball_vel_x = float(o[4]) if len(o) > 4 else 0.0

            shaped  = W_BALL_DIST * (np.sqrt(prev_ball_x**2 + prev_ball_y**2)
                                   - np.sqrt(ball_x**2      + ball_y**2))
            shaped += W_BALL_GOAL * max(0.0, ball_vel_x)
            shaped += W_SHOOT     * max(0.0, ball_vel_x - 0.3)
            if ball_x < 0:
                shaped += W_BLOCK

            teammate_ids = [a for a in agents if a != agent_id]
            for tid in teammate_ids:
                t_o = obs_map[tid]
                t_ball_x = float(t_o[7] if len(t_o) >= 8 else t_o[0])
                if ball_x * t_ball_x < 0:
                    shaped += W_SPREAD

            episode._agent_reward_history[agent_id][-1] += shaped

    def on_train_result(self, *, trainer, result, **kwargs):
        it         = result["training_iteration"]
        steps      = result["timesteps_total"]
        pct        = 100 * steps / TARGET_STEPS
        mean_reward = result.get("episode_reward_mean", float("nan"))
        ep_len     = result.get("episode_len_mean", float("nan"))
        eps        = result.get("episodes_total", 0)
        print(
            f"  iter {it:4d} | steps {steps:>9,} ({pct:5.1f}%) | "
            f"reward {mean_reward:+.4f} | ep_len {ep_len:6.1f} | eps {eps:5d}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint from phase 1 or 2 to restore from")
    args = parser.parse_args()

    for name in ["mlagents_envs", "ray", "ray.tune", "ray.rllib"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    ray.init(include_dashboard=False, num_cpus=24, log_to_driver=True,
             logging_level=logging.ERROR)

    tune.registry.register_env("Soccer", create_rllib_env)

    # Get obs/act spaces
    _tmp = create_rllib_env({"variation": EnvType.multiagent_team,
                              "base_port": 51500, "worker_id": 0})
    obs_space = _tmp.observation_space
    act_space = _tmp.action_space
    _tmp.close()

    print("=" * 65)
    print("  PHASE 3: Self-Play Fine-Tuning — Ray 1.4 / CPU")
    print(f"  Restoring from: {args.checkpoint}")
    print(f"  Workers : {NUM_WORKERS} x {NUM_ENVS_PER_WORKER}")
    print(f"  Target  : {TARGET_STEPS:,} steps")
    print("=" * 65)

    analysis = tune.run(
        "PPO",
        name="cpu_selfplay_phase3",
        restore=args.checkpoint,
        config={
            "num_gpus": 0,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "num_cpus_per_worker": 1,
            "log_level": "ERROR",
            "framework": "torch",
            "callbacks": SelfPlayCallback,
            "multiagent": {
                "policies": {"default": (None, obs_space, act_space, {})},
                "policy_mapping_fn": tune.function(lambda _: "default"),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_team,
                "base_port": 51000,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 500,
            "train_batch_size": TRAIN_BATCH,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 8,
            # Lower LR for fine-tuning
            "lr_schedule": [
                [0,            1e-4],
                [8_000_000,    5e-5],
                [12_000_000,   1e-5],
            ],
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.15,   # tighter clip to preserve pre-trained policy
            "entropy_coeff": 0.005,
            "kl_coeff": 0.3,
            "kl_target": 0.01,
        },
        stop={"timesteps_total": TARGET_STEPS},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray_results"),
        verbose=0,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print("\n" + "=" * 65)
    print(f"  Done! Best checkpoint: {best_checkpoint}")
    print("=" * 65)
