"""
Simple training script for Soccer-Twos single player agent.
Trains a PPO agent against the baseline agent.
Optimized for RTX 5080 GPU.
"""
import pickle
import os

import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls
from soccer_twos import EnvType

from utils import create_rllib_env


# Constants for baseline agent
ALGORITHM = "PPO"
BASELINE_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)
POLICY_NAME = "default"

# Cache the baseline trainer to avoid reloading
_baseline_trainer = None


def get_baseline_opponent():
    """Returns a baseline opponent that can be used by the environment.
    
    The opponent_policy should be a callable that takes observations (numpy array)
    and returns actions (numpy array).
    """
    global _baseline_trainer
    
    if _baseline_trainer is None:
        # Load config from checkpoint
        config_dir = os.path.dirname(BASELINE_CHECKPOINT)
        config_path = os.path.join(config_dir, "params.pkl")
        
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find params.pkl at {config_path}")
        
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        
        # Disable parallelism for evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0
        
        # Create dummy env for initialization
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"
        
        # Create and load the trainer
        cls = get_trainable_cls(ALGORITHM)
        _baseline_trainer = cls(env=config["env"], config=config)
        _baseline_trainer.restore(BASELINE_CHECKPOINT)
        _baseline_policy = _baseline_trainer.get_policy(POLICY_NAME)
        
        # Store policy reference
        _baseline_trainer._policy = _baseline_policy
    
    def opponent_fn(observation):
        """Takes a single observation and returns an action."""
        # compute_single_action returns (action, action_info, ...)
        action, *_ = _baseline_trainer._policy.compute_single_action(observation)
        return action
    
    return opponent_fn


NUM_ENVS_PER_WORKER = 4  # More envs per worker for faster sampling


if __name__ == "__main__":
    # Disable dashboard and metrics to avoid hostname issues
    import os
    os.environ["RAY_DASHBOARD_ENABLED"] = "0"
    os.environ["RAY_CLOUD_PLATFORM"] = "aws"  # Skip cloud detection
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    os.environ["RAY_EVENT_ENABLE_LEGACY_GCS_SERVICE"] = "1"
    os.environ["RAY_METRICS_ENABLE_GCS"] = "0"  # Disable GCS metrics
    
    ray.init(
        include_dashboard=False,
        dashboard_host="127.0.0.1",
        num_cpus=24,
        _metrics_export_port=None,  # Disable metrics port
    )

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_SP_vs_Baseline",
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
                "opponent_policy": get_baseline_opponent(),  # Use baseline!
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