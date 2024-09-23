# main.py
import argparse
import torch
from src.env import VLSIEnvironment
from src.agent import DQNAgent, select_action
from src.ppo_agent import train_ppo, evaluate_ppo
import torch.optim as optim
import torch.nn as nn

def train_model():
    # Hyperparameters
    num_episodes = 1000
    epsilon = 0.1
    gamma = 0.99
    learning_rate = 1e-4

    # Initialize environment and agent
    env = VLSIEnvironment()
    n_actions = env.action_space.n
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    
    policy_net = DQNAgent(input_dim, n_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for t in range(100):
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
            
            if done:
                break
            state = next_state
        
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")
        
        # Save the model periodically
        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), f"models/dqn_model_{episode + 1}.pth")
    
    # Save the final model
    torch.save(policy_net.state_dict(), "models/trained_dqn.pth")
    print("Training completed and model saved to models/trained_dqn.pth")

def evaluate_model():
    import matplotlib.pyplot as plt
    
    env = VLSIEnvironment()
    n_actions = env.action_space.n
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    
    # Load the trained model
    policy_net = DQNAgent(input_dim, n_actions)
    policy_net.load_state_dict(torch.load("models/trained_dqn.pth"))
    policy_net.eval()

    state = env.reset()
    done = False

    while not done:
        action = select_action(state, policy_net, epsilon=0, n_actions=n_actions)
        state, reward, done, _ = env.step(action)

    # Visualize the final layout
    plt.imshow(state, cmap='gray')
    plt.title("Final VLSI Floorplan")
    plt.show()
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLSI Floorplanning with Deep Reinforcement Learning")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True,
                    help="Mode of operation: 'train' or 'evaluate'")
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], default="dqn",
                    help="Choose the RL algorithm: 'dqn' or 'ppo'")
    
    args = parser.parse_args()
    
    if args.algo == "ppo" and args.mode == "train":
        train_ppo()
    elif args.algo == "ppo" and args.mode == "evaluate":
        evaluate_ppo()
    elif args.mode == "train":
        train_model()
    elif args.mode == "evaluate":
        evaluate_model()
