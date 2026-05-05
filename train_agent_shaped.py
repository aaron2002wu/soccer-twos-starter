"""
Team multiagent training with reward shaping + curriculum + self-play.
Ray 1.4 / soccertwos conda env — CPU only.

Your policy controls BOTH players on team 0 (agents 0 & 1).
This enables emergent passing, positioning, and coordinated defense.

Phases:
  1. Curriculum (stages 0-4): ball placement gets progressively harder
  2. vs Baseline: full game against CEIA baseline team
  3. Self-play: once reward > SELFPLAY_THRESHOLD, opponent pool updates
     to past versions of your own policy

Reward shaping per player:
  - Moving toward ball
  - Ball moving toward opponent goal
  - Blocking (being between ball and own goal)
  - Proximity bonus: reward when teammate is on opposite side of ball (passing setup)
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


# ── Hyperparameters ──────────────────────────────────────────────────────────
NUM_WORKERS          = 20
NUM_ENVS_PER_WORKER  = 1
TRAIN_BATCH          = 10000
TARGET_STEPS         = 20_000_000
ADVANCE_THRESHOLD    = 0.3
SELFPLAY_THRESHOLD   = 1.5

W_BALL_DIST   = 0.02   # reduced now agent is learning
W_BALL_GOAL   = 0.03
W_BLOCK       = 0.002
W_SPREAD      = 0.002
W_SHOOT       = 0.05   # reduced but still strongest signal
W_TEAMMATE    = 0.01


# ── Curriculum ───────────────────────────────────────────────────────────────
with open("curriculum.yaml") as f:
    _curriculum = yaml.load(f, Loader=yaml.FullLoader)

TASKS = _curriculum["tasks"] + [{
    "name": "Full Self-Play",
    "config_fn": "none",
    "ranges": {
        "ball": {
            "position": {"x": [-10, 14], "y": [-5, 5]},
            "velocity": {"x": [-10, 10], "y": [-10, 10]},
        },
        "players": {
            0: {"rotation_y": [0, 360], "position": {"x": [-14, 14], "y": [-5, 5]}},
            1: {"rotation_y": [0, 360], "position": {"x": [-14, 14], "y": [-5, 5]}},
        },
    },
}]

_current_task  = 0
_selfplay_mode = False
_selfplay_weights = {"pool": [None, None, None]}


# ── Baseline weights ─────────────────────────────────────────────────────────
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


def _make_opponent(weights_bytes_fallback):
    """
    Opponent callable for team_vs_policy — called once per team step.
    In self-play mode uses pool of past selves, otherwise uses baseline.
    Returns a MultiDiscrete action (not flattened) since multiagent=True.
    """
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

    def _act(obs, net):
        with torch.no_grad():
            t = torch.tensor(np.array(obs, dtype=np.float32)).unsqueeze(0)
            logits = net(t).squeeze(0)
            a0 = logits[0:3].argmax().item()
            a1 = logits[3:6].argmax().item()
            a2 = logits[6:9].argmax().item()
            return np.array([a0, a1, a2])

    def opponent_fn(obs):
        # obs is a dict {player_id: obs_array} for the opponent team
        pool = _selfplay_weights["pool"]
        candidates = [w for w in pool if w is not None]

        if candidates:
            weights = candidates[np.random.choice(
                len(candidates),
                p=([0.6, 0.3, 0.1] if len(candidates) == 3
                   else [0.7, 0.3] if len(candidates) == 2
                   else [1.0])
            )]
            key = id(weights)
        else:
            weights = None
            key = -1

        if not hasattr(opponent_fn, "_net") or opponent_fn._key != key:
            n = _Net()
            if weights is not None:
                n.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
            else:
                n.load_state_dict(torch.load(io.BytesIO(weights_bytes_fallback)))
            n.eval()
            opponent_fn._net = n
            opponent_fn._key = key

        if isinstance(obs, dict):
            return {pid: _act(o, opponent_fn._net) for pid, o in obs.items()}
        return _act(obs, opponent_fn._net)

    return opponent_fn


# ── Callbacks ────────────────────────────────────────────────────────────────
class TeamShapedCallback(DefaultCallbacks):

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
            except Exception as e:
                if not hasattr(self, '_warned'):
                    print(f"  [curriculum] env_channel unavailable: {e}", flush=True)
                    self._warned = True

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        if _current_task < 1:
            return

        agents = episode.get_agent_ids()
        obs_map = {a: episode.last_observation_for(a) for a in agents}
        prev_map = {a: episode._agent_to_last_obs.get(a) for a in agents}

        for agent_id in agents:
            obs = obs_map[agent_id]
            prev_obs = prev_map[agent_id]
            if prev_obs is None:
                continue

            shaped = 0.0

            # Each agent obs is 336-dim (or 672 stacked — use first 336)
            o = obs[:336] if len(obs) >= 336 else obs
            p = prev_obs[:336] if len(prev_obs) >= 336 else prev_obs

            # obs[0:3] = self forward/right/up (unit vectors, always ~1)
            # obs[7:9] = ball relative position (x, y normalized)
            # obs[4:6] = ball relative velocity
            ball_x, ball_y = float(o[7]), float(o[8])
            prev_ball_x, prev_ball_y = float(p[7]), float(p[8])

            # Move toward ball
            ball_dist_now  = np.sqrt(ball_x**2 + ball_y**2)
            ball_dist_prev = np.sqrt(prev_ball_x**2 + prev_ball_y**2)
            shaped += W_BALL_DIST * (ball_dist_prev - ball_dist_now)

            # Ball velocity toward opponent goal (positive x = toward opp goal)
            ball_vel_x = float(o[4]) if len(o) > 4 else 0.0
            shaped += W_BALL_GOAL * max(0.0, ball_vel_x)

            # Strong shoot reward: ball moving fast in goal direction
            if ball_vel_x > 0.3:
                shaped += W_SHOOT * ball_vel_x

            # Blocking: agent between ball and own goal (ball_x < 0 = in our half)
            if ball_x < 0:
                shaped += W_BLOCK

            # Spread: reward teammates being on opposite sides of ball
            teammate_ids = [a for a in agents if a != agent_id]
            for tid in teammate_ids:
                t_obs = obs_map[tid]
                t_o = t_obs[:336] if len(t_obs) >= 336 else t_obs
                t_ball_x = float(t_o[7])
                if ball_x * t_ball_x < 0:  # opposite sides of ball
                    shaped += W_SPREAD

            episode._agent_reward_history[agent_id][-1] += shaped

    def on_train_result(self, *, trainer, result, **kwargs):
        global _current_task, _selfplay_mode
        mean_reward = result.get("episode_reward_mean", float("nan"))
        ep_len = result.get("episode_len_mean", 1000.0)
        it    = result["training_iteration"]
        steps = result["timesteps_total"]
        pct   = 100 * steps / TARGET_STEPS
        eps   = result.get("episodes_total", 0)

        # Curriculum: advance when ep_len drops below threshold for current stage
        # Thresholds set below starting ep_len (~525) so stages are meaningful
        ep_len_threshold = 480 - (_current_task * 30)
        if ep_len < ep_len_threshold and _current_task < len(TASKS) - 1:
            old = TASKS[_current_task]["name"]
            _current_task += 1
            print(f"\n  *** ADVANCE: '{old}' -> '{TASKS[_current_task]['name']}' "
                  f"(ep_len {ep_len:.0f} < {ep_len_threshold}) ***\n", flush=True)

        mode_str = f"stage[{_current_task}] {TASKS[_current_task]['name']}"
        print(
            f"  iter {it:4d} | steps {steps:>9,} ({pct:5.1f}%) | "
            f"{mode_str:<28} | reward {mean_reward:+.4f} | "
            f"ep_len {ep_len:6.1f} | eps {eps:5d}",
            flush=True
        )


if __name__ == "__main__":
    for name in ["mlagents_envs", "ray", "ray.tune", "ray.rllib"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    ray.init(include_dashboard=False, num_cpus=24, log_to_driver=True,
             logging_level=logging.ERROR)

    tune.registry.register_env("Soccer", create_rllib_env)

    # Get spaces from a temp env (team_vs_policy, both players, not flattened)
    _tmp = create_rllib_env({"variation": EnvType.multiagent_team, "base_port": 51500, "worker_id": 0})
    obs_space = _tmp.observation_space
    act_space = _tmp.action_space
    _tmp.close()

    print("=" * 65)
    print("  TEAM SHAPED CURRICULUM + SELF-PLAY — Ray 1.4 / CPU")
    print("  Controls both players on team (enables passing/coordination)")
    for i, t in enumerate(TASKS):
        print(f"    Stage {i}: {t['name']}")
    print(f"\n  Curriculum advance : ep_len decreasing (950→880→810...)")
    print(f"  Self-play          : multiagent_team (always active)")
    print(f"  Workers            : {NUM_WORKERS} x {NUM_ENVS_PER_WORKER}")
    print(f"  Target steps       : {TARGET_STEPS:,}")
    print("=" * 65)

    analysis = tune.run(
        "PPO",
        name="cpu_team_selfplay_run1",
        config={
            "num_gpus": 0,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "num_cpus_per_worker": 1,
            "log_level": "ERROR",
            "framework": "torch",
            "callbacks": TeamShapedCallback,
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
            "lr_schedule": [
                [0,            3e-4],
                [8_000_000,    1e-4],
                [14_000_000,   5e-5],
            ],
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.02,
            "entropy_coeff_schedule": [
                [0,            0.02],
                [8_000_000,    0.01],
                [14_000_000,   0.005],
            ],
            "kl_coeff": 0.2,
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
