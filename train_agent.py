"""
Simple training script for Soccer-Twos single player agent.
Trains a PPO agent against the CEIA baseline opponent.
Optimized for RTX 5070 GPU with Ray 2.x / PyTorch 2.x.
"""
import os
import logging

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from soccer_twos import EnvType

from utils import create_rllib_env
from baseline_policy import load_baseline_policy


NUM_ENVS_PER_WORKER = 2
NUM_WORKERS = 22


if __name__ == "__main__":
    os.environ["RAY_DASHBOARD_ENABLED"] = "0"
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

    logging.getLogger("mlagents_envs").setLevel(logging.WARNING)
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.WARNING)

    ray.init(include_dashboard=False, num_cpus=24)

    register_env("Soccer", create_rllib_env)

    # Load baseline policy once — shared across all workers via closure
    baseline_policy = load_baseline_policy()

    print("=" * 60)
    print("Starting PPO training vs CEIA baseline opponent")
    print(f"  Workers: {NUM_WORKERS} x {NUM_ENVS_PER_WORKER} envs = {NUM_WORKERS * NUM_ENVS_PER_WORKER} Unity processes")
    print(f"  GPU: 1  |  Batch: 20000  |  Target: 10M steps")
    print("=" * 60)

    config = (
        PPOConfig()
        .environment(
            env="Soccer",
            disable_env_checking=True,
            env_config={
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": baseline_policy,
                "base_port": 50500,
            },
        )
        .framework("torch")
        .resources(num_gpus=1)
        .rollouts(
            num_rollout_workers=NUM_WORKERS,
            num_envs_per_worker=NUM_ENVS_PER_WORKER,
            rollout_fragment_length=500,
        )
        .training(
            train_batch_size=20000,
            sgd_minibatch_size=512,
            num_sgd_iter=8,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            model={
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512, 256],
            },
        )
        .debugging(log_level="WARN")
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name="trainRun1",
            stop={"timesteps_total": 10_000_000},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=100,
                checkpoint_at_end=True,
            ),
            local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray_results"),
            verbose=2,
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best trial config: {best.config}")
    print(f"Best checkpoint: {best.checkpoint}")
    print("Done training!")
