import torch
import numpy as np
from space_rocks_env import SpaceRocksEnv
from dqn_agent import DQNAgent

def preprocess_observation(obs):
    # Normalize pixel values and resize if necessary
    obs = obs / 255.0
    return obs

def train():
    env = SpaceRocksEnv(render_mode=False)
    device = torch.device("cpu")
    input_shape = (3, 600, 800)  # Channels, Height, Width
    num_actions = env.action_space.n
    agent = DQNAgent(input_shape, num_actions, device)

    num_episodes = 1000
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = preprocess_observation(obs)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = preprocess_observation(next_obs)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")

    env.close()

if __name__ == "__main__":
    train()
