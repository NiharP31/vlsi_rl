import gym
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env import VLSIEnvironment

def train_ppo():
    # Wrap the VLSI environment
    env = DummyVecEnv([lambda: VLSIEnvironment()])
    
    # Initialize the PPO model
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, gamma=0.99, n_steps=2048, batch_size=64, n_epochs=10)

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("models/ppo_vlsi")

def evaluate_ppo():
    env = VLSIEnvironment()
    model = PPO.load("models/ppo_vlsi")

    state = env.reset()
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, _ = env.step(action)

        # Display or visualize the state
        env.render()
