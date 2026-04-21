"""
Loads the CEIA baseline agent's policy weights directly via PyTorch,
bypassing the old Ray 1.x checkpoint API.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn


CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos"
    "/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    "/checkpoint_002449/checkpoint-2449",
)


class FCNet(nn.Module):
    """Matches the baseline: fcnet_hiddens=[256,256], MultiDiscrete([3,3,3]) output."""
    def __init__(self, obs_size=336, hidden=[256, 256]):
        super().__init__()
        layers = []
        in_size = obs_size
        for h in hidden:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        self.shared = nn.Sequential(*layers)
        # 3 separate action heads, each size 3 (MultiDiscrete [3,3,3])
        self.action_head = nn.Linear(in_size, 9)  # Ray flattens to 9 logits total

    def forward(self, x):
        return self.action_head(self.shared(x))


def load_baseline_policy():
    """
    Loads weights from the Ray 1.x checkpoint pickle and returns
    a callable policy: obs (np.ndarray) -> action (int).
    """
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint_data = pickle.load(f)

    # Ray 1.x stores weights directly under state["default"]
    worker_data = pickle.loads(checkpoint_data["worker"])
    weights = worker_data["state"]["default"]  # keys are the weight tensor names

    net = FCNet()

    # Map flat weight arrays from checkpoint into the model
    ray_weights = {k: torch.tensor(np.array(v)) for k, v in weights.items()
                   if not k.startswith("_optimizer")}

    mapping = {
        "shared.0.weight": "_hidden_layers.0._model.0.weight",
        "shared.0.bias":   "_hidden_layers.0._model.0.bias",
        "shared.2.weight": "_hidden_layers.1._model.0.weight",
        "shared.2.bias":   "_hidden_layers.1._model.0.bias",
        "action_head.weight": "_logits._model.0.weight",
        "action_head.bias":   "_logits._model.0.bias",
    }

    new_state = {}
    for our_key, ray_key in mapping.items():
        if ray_key in ray_weights:
            new_state[our_key] = ray_weights[ray_key]
        else:
            raise KeyError(f"Expected key '{ray_key}' not found in checkpoint. "
                           f"Available: {list(ray_weights.keys())}")

    net.load_state_dict(new_state)
    net.eval()

    def policy(obs):
        """Returns a flattened action int (0-26) compatible with flatten_branched=True."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = net(obs_t).squeeze(0)  # shape [9]
            # Split into 3 branches of 3, argmax each
            a0 = logits[0:3].argmax().item()
            a1 = logits[3:6].argmax().item()
            a2 = logits[6:9].argmax().item()
            # Flatten MultiDiscrete([3,3,3]) -> single int
            action = a0 * 9 + a1 * 3 + a2
        return action

    return policy


if __name__ == "__main__":
    # Quick sanity check
    policy = load_baseline_policy()
    dummy_obs = np.zeros(336, dtype=np.float32)
    action = policy(dummy_obs)
    print(f"Baseline policy loaded OK. Test action: {action}")
