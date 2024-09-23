# src/train.py
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from env import VLSIEnvironment
from agent import DQNAgent, select_action
from data_aug import augment_data

# Hyperparameters
num_episodes = 1000
epsilon = 0.1
gamma = 0.99
learning_rate = 1e-4
initial_epsilon = 1.0  # Start with a high epsilon for more exploration
min_epsilon = 0.01     # Minimum value of epsilon to allow some exploration
epsilon_decay = 0.995  # Decay rate

env = VLSIEnvironment()
n_actions = env.action_space.n
input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]

policy_net = DQNAgent(input_dim, n_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
criterion = nn.MSELoss()

for episode in range(num_episodes):
    epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** episode))  # Decay epsilon
    state = env.reset()
    total_reward = 0
    
    for t in range(100):
        state = augment_data(state)
        action = select_action(state, policy_net, epsilon, n_actions)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Compute target
        target = reward + gamma * policy_net(torch.FloatTensor(next_state.flatten()).unsqueeze(0)).max().item()
        prediction = policy_net(torch.FloatTensor(state.flatten()).unsqueeze(0))[0, action]
        
        # Update the network
        loss = criterion(prediction, torch.FloatTensor([target]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        
        if done:
            break
        state = next_state
    
    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")
