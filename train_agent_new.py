"""
Curriculum training script for Soccer-Twos.
Ray 1.4 / soccertwos conda env — CPU only.

Stages (from curriculum.yaml):
  0 - Very Easy Goal  : ball placed near goal, agent close & facing it
  1 - Easy Goal       : ball in attacking half, random velocity
  2 - Medium Goal     : ball anywhere, agent in attacking half
  3 - Hard Goal       : ball anywhere, agent anywhere
  4 - Random Players  : full game, opponents move randomly
  5 - vs Baseline     : full game vs CEIA baseline

Advances to next stage when episode_reward_mean > ADVANCE_THRESHOLD.
"""
import os
import io
import logging
import warnings
import pickle
import yaml

os.environ["RAY_DASHBOARD_ENABLED"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env, sample_pos_vel, sample_player


NUM_WORKERS = 10
NUM_ENVS_PER_WORKER = 1
TRAIN_BATCH = 10000
TARGET_STEPS = 10_000_000
ADVANCE_THRESHOLD = 0.5


# ── Load curriculum ──────────────────────────────────────────────────────────
with open("curriculum.yaml") as f:
    _curriculum = yaml.load(f, Loader=yaml.FullLoader)

TASKS = _curriculum["tasks"] + [{
    "name": "vs Baseline",
    "config_fn": "baseline",
    "ranges": {
        "ball": {
            "position": {"x": [-10, 14], "y": [-5, 5]},
            "velocity": {"x": [-10, 10], "y": [-10, 10]},
        },
        "players": {
            0: {"rotation_y": [0, 360], "position": {"x": [-14, 14], "y": [-5, 5]}},
        },
    },
}]

_current_task = 0


# ── Load baseline weights into a picklable bytes blob ───────────────────────
BASELINE_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos"
    "/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    "/checkpoint_002449/checkpoint-2449",
)

def _load_baseline_weights_bytes():
    with open(BASELINE_CHECKPOINT, "rb") as f:
        data = pickle.load(f)
    w = pickle.loads(data["worker"])["state"]["default"]

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
        "shared.0.weight": torch.tensor(np.array(w["_hidden_layers.0._model.0.weight"])),
        "shared.0.bias":   torch.tensor(np.array(w["_hidden_layers.0._model.0.bias"])),
        "shared.2.weight": torch.tensor(np.array(w["_hidden_layers.1._model.0.weight"])),
        "shared.2.bias":   torch.tensor(np.array(w["_hidden_layers.1._model.0.bias"])),
        "head.weight":     torch.tensor(np.array(w["_logits._model.0.weight"])),
        "head.bias":       torch.tensor(np.array(w["_logits._model.0.bias"])),
    })
    net.eval()
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()

_BASELINE_WEIGHTS_BYTES = _load_baseline_weights_bytes()


def _make_baseline_opponent(weights_bytes):
    """Returns a picklable opponent callable using raw weight bytes."""
    def opponent_fn(obs):
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


# ── Callbacks ────────────────────────────────────────────────────────────────
class CurriculumCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        task = TASKS[_current_task]
        for env in base_env.get_unwrapped():
            try:
                env.env_channel.set_parameters(
                    ball_state=sample_pos_vel(task["ranges"]["ball"]),
                    players_states={
                        p: sample_player(task["ranges"]["players"][p])
                        for p in task["ranges"]["players"]
                    },
                )
            except Exception:
                pass

    def on_train_result(self, *, trainer, result, **kwargs):
        global _current_task
        mean_reward = result.get("episode_reward_mean", float("nan"))
        it = result["training_iteration"]
        steps = result["timesteps_total"]
        pct = 100 * steps / TARGET_STEPS
        current_name = TASKS[_current_task]["name"]

        if mean_reward > ADVANCE_THRESHOLD and _current_task < len(TASKS) - 1:
            _current_task += 1
            next_name = TASKS[_current_task]["name"]
            print(f"\n  *** ADVANCE: '{current_name}' -> '{next_name}' "
                  f"(reward {mean_reward:.3f}) ***\n")
        else:
            ep_len = result.get("episode_len_mean", float("nan"))
            eps = result.get("episodes_total", 0)
            print(
                f"  iter {it:4d} | steps {steps:>9,} ({pct:5.1f}%) | "
                f"stage [{_current_task}] {current_name:<18} | "
                f"reward {mean_reward:+.4f} | ep_len {ep_len:6.1f} | eps {eps:5d}"
            )


if __name__ == "__main__":
    for name in ["mlagents_envs", "ray", "ray.tune", "ray.rllib"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    ray.init(include_dashboard=False, num_cpus=24, log_to_driver=False,
             logging_level=logging.ERROR)

    tune.registry.register_env("Soccer", create_rllib_env)

    print("=" * 65)
    print("  CURRICULUM Training — Ray 1.4 / CPU / vs CEIA Baseline")
    for i, t in enumerate(TASKS):
        print(f"    Stage {i}: {t['name']}")
    print(f"\n  Advance threshold : reward > {ADVANCE_THRESHOLD}")
    print(f"  Workers           : {NUM_WORKERS} x {NUM_ENVS_PER_WORKER}")
    print(f"  Target steps      : {TARGET_STEPS:,}")
    print("=" * 65)

    baseline_opponent = _make_baseline_opponent(_BASELINE_WEIGHTS_BYTES)

    analysis = tune.run(
        "PPO",
        name="cpu_curriculum_run1",
        config={
            "num_gpus": 0,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "num_cpus_per_worker": 1,
            "log_level": "ERROR",
            "framework": "torch",
            "callbacks": CurriculumCallback,
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": baseline_opponent,
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
        verbose=0,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print("\n" + "=" * 65)
    print(f"  Done! Best checkpoint: {best_checkpoint}")
    print("=" * 65)
