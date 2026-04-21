"""
Simple training script for Soccer-Twos single player agent.
Trains a PPO agent against a stationary opponent.
Optimized for RTX 5080 GPU.
"""
import os
import logging

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 2


if __name__ == "__main__":
    # Suppress Ray dashboard/metrics noise
    os.environ["RAY_DASHBOARD_ENABLED"] = "0"
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    os.environ["RAY_METRICS_ENABLE_GCS"] = "0"

    logging.getLogger("mlagents_envs").setLevel(logging.WARNING)
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.WARNING)

    ray.init(include_dashboard=False, num_cpus=24, _metrics_export_port=None)

    tune.registry.register_env("Soccer", create_rllib_env)

    print("=" * 60)
    print("Starting PPO training vs stationary opponent")
    print(f"  Workers: 22 x {NUM_ENVS_PER_WORKER} envs = {22 * NUM_ENVS_PER_WORKER} Unity processes")
    print(f"  GPU: 1  |  Batch: 20000  |  Target: 10M steps")
    print("=" * 60)

    analysis = tune.run(
        "PPO",
        name="PPO_test",
        config={
            "num_gpus": 1,
            "num_workers": 22,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "WARN",
            "framework": "torch",
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,  # stationary opponent
                "base_port": 50500,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512, 256],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 20000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 8,
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
        },
        stop={"timesteps_total": 10000000},
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        verbose=2,
        progress_reporter=tune.CLIReporter(
            metric_columns=["episode_reward_mean", "timesteps_total", "training_iteration"],
            print_intermediate_tables=True,
        ),
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(f"Best trial: {best_trial}")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(f"Best checkpoint: {best_checkpoint}")
    print("Done training!")
