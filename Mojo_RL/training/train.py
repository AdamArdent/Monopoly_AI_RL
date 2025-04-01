# training/train.py

import numpy as np
from Mojo_RL.env.mojo_env import MojoEnv
from Mojo_RL.agents.agent import Agent

def train_agent(num_episodes=500):
    env = MojoEnv()
    agent = Agent(env.action_space, env.observation_space)
    rewards_history = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        counter = 0
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            counter += 1
            print(f"Reward : {reward} for step {counter}")
        rewards_history.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards_history

# Pour permettre l'exécution directe de ce script
if __name__ == "__main__":
    train_agent()
