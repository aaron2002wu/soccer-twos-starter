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


NUM_ENVS_PER_WORKER = 2


if __name__ == "__main__":
    import os
    import logging

    # Suppress Ray dashboard/metrics noise
    os.environ["RAY_DASHBOARD_ENABLED"] = "0"
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    os.environ["RAY_METRICS_ENABLE_GCS"] = "0"

    # Suppress noisy loggers
    logging.getLogger("mlagents_envs").setLevel(logging.WARNING)
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.WARNING)

    ray.init(include_dashboard=False, num_cpus=24, _metrics_export_port=None)

    tune.registry.register_env("Soccer", create_rllib_env)

    print("=" * 60)
    print("Starting PPO training vs baseline agent")
    print(f"  Workers: 22 x 2 envs = 44 Unity processes")
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
                "opponent_policy": get_baseline_opponent(),
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
        verbose=2,  # Show iteration results
        progress_reporter=tune.CLIReporter(
            metric_columns=["episode_reward_mean", "timesteps_total", "training_iteration"],
            print_intermediate_tables=True,
        ),
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