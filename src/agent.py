import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def select_action(state, policy_net, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)  # Explore: select random action
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
            return policy_net(state_tensor).argmax().item()  # Exploit: select best action
