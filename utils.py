from random import uniform as randfloat

import gym
import gymnasium
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """
    pass


class GymV21CompatWrapper(gymnasium.Env):
    """
    Wraps an old gym-style env to expose the new gymnasium API
    (reset returns (obs, info), step returns (obs, reward, terminated, truncated, info)).
    Also explicitly exposes observation_space and action_space so Ray can find them.
    """
    def __init__(self, env):
        self._env = env
        self.observation_space = gymnasium.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(env.action_space.n)

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        result = self._env.step(action)
        obs, reward, done, info = result[0], result[1], result[2], result[-1]
        return np.array(obs, dtype=np.float32), float(reward), done, False, info

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    """
    if hasattr(env_config, "worker_index"):
        # worker_index is 1-based in Ray 2.x; subtract 1 to make it 0-based
        worker_index = env_config.worker_index - 1
        env_config["worker_id"] = (
            worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return GymV21CompatWrapper(env)
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
