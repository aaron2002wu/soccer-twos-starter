from typing import Dict
import os

import numpy as np
import gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from soccer_twos import AgentInterface, EnvType


# Point this to your checkpoint folder
CHECKPOINT_PATH = os.path.expanduser(
    "~/ray_results/PPO_test/PPO_Soccer_743d7_00000_0_2026-04-21_15-51-14/checkpoint_000004"
)

# Must match what you trained with
MODEL_CONFIG = {
    "vf_share_layers": True,
    "fcnet_hiddens": [512, 512, 256],
}


def _find_latest_checkpoint(base_path):
    """Walk the run folder and return the latest checkpoint path."""
    import glob
    # Ray 2.x checkpoint folders are named checkpoint_XXXXXX
    checkpoints = sorted(glob.glob(os.path.join(base_path, "**/checkpoint_*"), recursive=True))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {base_path}")
    return checkpoints[-1]


class MyAgent(AgentInterface):
    """
    Trained PPO agent loaded from a Ray 2.x checkpoint.
    Trained with flatten_branched=True so actions are Discrete(27).
    act() converts back to MultiDiscrete([3,3,3]) for the watch environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__()

        checkpoint_path = CHECKPOINT_PATH
        print(f"[MyAgent] Loading checkpoint: {checkpoint_path}")

        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1)

        import gymnasium
        obs_space = gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(336,), dtype=np.float32)
        act_space = gymnasium.spaces.Discrete(27)

        config = (
            PPOConfig()
            .environment(
                observation_space=obs_space,
                action_space=act_space,
                disable_env_checking=True,
            )
            .framework("torch")
            .resources(num_gpus=0)
            .rollouts(num_rollout_workers=0)
            .training(model=MODEL_CONFIG)
        )

        algo = config.build()
        algo.restore(checkpoint_path)
        self.policy = algo.get_policy()

        # MultiDiscrete([3,3,3]) branch sizes for unflattening
        self._branch_sizes = [3, 3, 3]

    def _unflatten(self, flat_action: int) -> np.ndarray:
        """Convert Discrete(27) int back to MultiDiscrete([3,3,3]) array."""
        actions = []
        for size in reversed(self._branch_sizes):
            actions.append(flat_action % size)
            flat_action //= size
        return np.array(list(reversed(actions)), dtype=np.int32)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            flat_action, _, _ = self.policy.compute_single_action(
                np.array(obs, dtype=np.float32)
            )
            actions[player_id] = self._unflatten(int(flat_action))
        return actions
