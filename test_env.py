"""
Quick test to verify the environment works.
Run this first to make sure everything is set up correctly.
"""
import soccer_twos
from soccer_twos import EnvType

# Create environment
env = soccer_twos.make(
    variation=EnvType.team_vs_policy,
    single_player=True,
    flatten_branched=True,
    opponent_policy=lambda *_: 0,
)

print("Environment created successfully!")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Run a few episodes
for episode in range(3):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    
    print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

env.close()
print("\nEnvironment test complete!")