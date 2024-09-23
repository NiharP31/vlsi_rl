import torch
import matplotlib.pyplot as plt
from env import VLSIEnvironment
from agent import DQNAgent

# Load the trained model
policy_net = DQNAgent(input_dim, n_actions)
policy_net.load_state_dict(torch.load("models/trained_dqn.pth"))

env = VLSIEnvironment()
state = env.reset()
done = False

while not done:
    action = select_action(state, policy_net, epsilon=0, n_actions=n_actions)
    state, reward, done, _ = env.step(action)

# Visualize the final layout
plt.imshow(state, cmap='gray')
plt.title("Final VLSI Floorplan")
plt.show()
