"""
Simple training script for Soccer-Twos single player agent.
Trains a PPO agent against a random opponent.
Optimized for RTX 5080 GPU.
"""
import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 4  # More envs per worker for faster sampling


if __name__ == "__main__":
    # Disable dashboard to avoid hostname issues
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_SP",
        config={
            # system settings - RTX 5080 optimized
            "num_gpus": 1,  # Use your GPU!
            "num_workers": 6,  # Parallel workers (adjust based on CPU cores)
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
                "base_port": 50500,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512, 256],  # Larger network for GPU
            },
            # Training parameters - optimized for speed
            "rollout_fragment_length": 500,
            "train_batch_size": 20000,  # Larger batch for GPU
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 8,
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
        },
        stop={
            "timesteps_total": 10000000,  # 10M steps - GPU can handle more
        },
        checkpoint_freq=100,
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