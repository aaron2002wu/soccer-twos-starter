"""
Test a trained agent by watching it play against a random opponent.
Run this locally after copying the trained checkpoint from the remote.
"""
import argparse
import gym
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
import soccer_twos
from soccer_twos import EnvType


def create_env():
    """Create the soccer environment."""
    return soccer_twos.make(
        variation=EnvType.team_vs_policy,
        single_player=True,
        flatten_branched=True,
        opponent_policy=lambda *_: 0,  # Random opponent
    )


def test_agent(checkpoint_path, num_episodes=5):
    """Test a trained agent."""
    # Load the trained agent
    config = {
        "env": "Soccer",
        "env_config": {
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": True,
            "flatten_branched": True,
            "opponent_policy": lambda *_: 0,
        },
        "framework": "torch",
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
        },
    }
    
    # Create the trainer and restore from checkpoint
    trainer = PPOTrainer(config=config, env="Soccer")
    trainer.restore(checkpoint_path)
    
    # Create environment
    env = create_env()
    
    # Run episodes
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # Get action from trained policy
            action = trainer.compute_action(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Render (if available)
            env.render()
        
        print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {step}")
    
    env.close()
    print("Testing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the trained checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to test")
    args = parser.parse_args()
    
    test_agent(args.checkpoint, args.episodes)