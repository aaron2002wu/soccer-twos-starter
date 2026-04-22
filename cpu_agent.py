"""
CPU training script for Soccer-Twos — compatible with Ray 1.4 / soccertwos conda env.
Trains a PPO agent against a stationary opponent on CPU only.
Maximizes 24-core usage: 20 rollout workers x 1 env each = 20 Unity processes.
"""
import os
import logging
import warnings

# Suppress Ray/gym noise before any imports
os.environ["RAY_DASHBOARD_ENABLED"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env


BASELINE_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos"
    "/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    "/checkpoint_002449/checkpoint-2449",
)


def load_baseline_opponent():
    """Load the CEIA baseline weights and return a picklable callable."""
    import pickle
    import numpy as np
    import torch
    import torch.nn as nn

    config_path = os.path.join(os.path.dirname(BASELINE_CHECKPOINT), "../params.pkl")
    with open(BASELINE_CHECKPOINT, "rb") as f:
        checkpoint_data = pickle.load(f)

    worker_data = pickle.loads(checkpoint_data["worker"])
    weights = worker_data["state"]["default"]

    # Build a plain PyTorch net — no Ray objects, fully picklable
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(336, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
            self.head = nn.Linear(256, 9)

        def forward(self, x):
            return self.head(self.shared(x))

    net = _Net()
    net.load_state_dict({
        "shared.0.weight": torch.tensor(np.array(weights["_hidden_layers.0._model.0.weight"])),
        "shared.0.bias":   torch.tensor(np.array(weights["_hidden_layers.0._model.0.bias"])),
        "shared.2.weight": torch.tensor(np.array(weights["_hidden_layers.1._model.0.weight"])),
        "shared.2.bias":   torch.tensor(np.array(weights["_hidden_layers.1._model.0.bias"])),
        "head.weight":     torch.tensor(np.array(weights["_logits._model.0.weight"])),
        "head.bias":       torch.tensor(np.array(weights["_logits._model.0.bias"])),
    })
    net.eval()

    # Serialize weights to bytes so the closure is pickle-safe
    import io
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    weights_bytes = buf.getvalue()

    def opponent_fn(obs):
        import io, torch, numpy as np
        import torch.nn as nn

        # Lazy-load on first call inside worker
        if not hasattr(opponent_fn, "_net"):
            class _Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.shared = nn.Sequential(
                        nn.Linear(336, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                    )
                    self.head = nn.Linear(256, 9)
                def forward(self, x):
                    return self.head(self.shared(x))
            n = _Net()
            n.load_state_dict(torch.load(io.BytesIO(weights_bytes)))
            n.eval()
            opponent_fn._net = n

        with torch.no_grad():
            t = torch.tensor(np.array(obs, dtype=np.float32)).unsqueeze(0)
            logits = opponent_fn._net(t).squeeze(0)
            a0 = logits[0:3].argmax().item()
            a1 = logits[3:6].argmax().item()
            a2 = logits[6:9].argmax().item()
            return a0 * 9 + a1 * 3 + a2

    return opponent_fn

from utils import create_rllib_env


NUM_WORKERS = 10          # split cores with curriculum run
NUM_ENVS_PER_WORKER = 1
TRAIN_BATCH = 10000
TARGET_STEPS = 5_000_000


class StatsCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs):
        it = result["training_iteration"]
        steps = result["timesteps_total"]
        reward = result.get("episode_reward_mean", float("nan"))
        ep_len = result.get("episode_len_mean", float("nan"))
        eps = result.get("episodes_total", 0)
        pct = 100 * steps / TARGET_STEPS
        print(
            f"  iter {it:4d} | steps {steps:>9,} ({pct:5.1f}%) | "
            f"reward {reward:+.4f} | ep_len {ep_len:6.1f} | episodes {eps:5d}"
        )


if __name__ == "__main__":
    # Silence everything
    for logger_name in ["mlagents_envs", "ray", "ray.tune", "ray.rllib"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    ray.init(
        include_dashboard=False,
        num_cpus=24,
        log_to_driver=False,
        logging_level=logging.ERROR,
    )

    tune.registry.register_env("Soccer", create_rllib_env)

    print("Loading CEIA baseline opponent...")
    baseline_opponent = load_baseline_opponent()
    print("Baseline loaded.")

    print("=" * 65)
    print("  CPU PPO Training vs CEIA Baseline — Ray 1.4 / soccertwos env")
    print(f"  Workers : {NUM_WORKERS} x {NUM_ENVS_PER_WORKER} = {NUM_WORKERS * NUM_ENVS_PER_WORKER} Unity processes")
    print(f"  Batch   : {TRAIN_BATCH:,} steps  |  Target: {TARGET_STEPS:,} steps")
    print("=" * 65)
    print(f"  {'iter':>4}  {'steps':>12}  {'%':>6}  {'reward':>8}  {'ep_len':>7}  {'episodes':>8}")
    print("  " + "-" * 60)

    analysis = tune.run(
        "PPO",
        name="cpu_train_run1",
        config={
            "num_gpus": 0,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "num_cpus_per_worker": 1,
            "log_level": "ERROR",
            "framework": "torch",
            "callbacks": StatsCallback,
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": baseline_opponent,
                "base_port": 50500,
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
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
        },
        stop={"timesteps_total": TARGET_STEPS},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray_results"),
        verbose=0,  # suppress tune's own output, we use callback instead
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print("\n" + "=" * 65)
    print(f"  Done! Best checkpoint: {best_checkpoint}")
    print("=" * 65)
