"""
Simple training script for Soccer-Twos single player agent.
Trains a PPO agent against a random opponent.
"""
import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1  # Keep low to avoid port conflicts


if __name__ == "__main__":
    # Disable dashboard to avoid hostname issues
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_SP",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 1,  # Keep low for desktop
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
                "base_port": 50500,  # Use high port to avoid conflicts
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
            },
            "rollout_fragment_length": 400,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
        },
        stop={
            "timesteps_total": 5000000,  # 5M steps - adjust as needed
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(f"Best trial: {best_trial}")
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(f"Best checkpoint: {best_checkpoint}")
    print("Done training!")