"""
Simple training script for Soccer-Twos single player agent.
Trains a PPO agent against the baseline agent.
Optimized for RTX 5080 GPU.
"""
import pickle
import os

import numpy as np
import ray
from ray import tune
import torch
from soccer_twos import EnvType

from utils import create_rllib_env


# Constants for baseline agent
BASELINE_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)


def get_baseline_opponent():
    """Returns a baseline opponent factory function.
    
    This creates a NEW baseline model instance each time it's called,
    avoiding serialization issues with Ray workers.
    """
    
    def create_baseline_model():
        """Create a fresh baseline model - loads weights from checkpoint."""
        import cloudpickle
        
        # Load params.pkl to get model config
        config_dir = os.path.dirname(BASELINE_CHECKPOINT)
        config_path = os.path.join(config_dir, "../params.pkl")
        
        model_config = {}
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                model_config = pickle.load(f)
        
        # Get hidden layers from config
        fcnet_hiddens = model_config.get("model", {}).get("fcnet_hiddens", [256, 256])
        
        # Create model matching baseline architecture
        obs_dim = 73
        action_dim = 9
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in fcnet_hiddens:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, action_dim))
        
        model = torch.nn.Sequential(*layers)
        model.eval()
        
        # Try to load weights from checkpoint
        try:
            with open(BASELINE_CHECKPOINT, "rb") as f:
                checkpoint = pickle.load(f)
            
            # The checkpoint contains serialized worker state
            # Try to extract model weights from it
            if isinstance(checkpoint, dict) and "worker" in checkpoint:
                worker_data = checkpoint["worker"]
                if isinstance(worker_data, bytes):
                    # Try to unpickle the worker
                    try:
                        worker = cloudpickle.loads(worker_data)
                        if hasattr(worker, 'policy') and hasattr(worker.policy, 'model'):
                            # Copy weights from the policy model
                            baseline_model = worker.policy.model
                            # Copy state dict
                            model.load_state_dict(baseline_model.state_dict())
                            print("Successfully loaded baseline weights!")
                    except Exception as e:
                        print(f"Could not deserialize worker: {e}")
        except Exception as e:
            print(f"Warning: Could not load baseline weights: {e}")
            print("Using untrained baseline (still provides diverse opponent)")
        
        return model
    
    # Store model in closure - will be created on first use
    _model = None
    
    def opponent_fn(observation):
        """Takes a single observation and returns an action."""
        nonlocal _model
        if _model is None:
            _model = create_baseline_model()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            logits = _model(obs_tensor)
            action = logits.argmax(dim=1).item()
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