import gym
import numpy as np

class VLSIEnvironment(gym.Env):
    def __init__(self, grid_size=(20, 20), num_blocks=10):
        super(VLSIEnvironment, self).__init__()
        
        # Define the grid (floorplan) size
        self.grid_size = grid_size
        self.num_blocks = num_blocks
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(grid_size[0] * grid_size[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.float32
        )
        
        # Initialize environment
        self.reset()

    def reset(self):
        # Create an empty grid
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        self.placed_blocks = []
        return self.grid

    def step(self, action):
        # Convert action to (x, y) coordinates
        x, y = divmod(action, self.grid_size[1])
        
        # Check if placement is valid
        if (x, y) in self.placed_blocks:
            reward = -1  # Penalize invalid actions
            done = True
        else:
            # Place a block on the grid
            self.grid[x, y] = 1
            self.placed_blocks.append((x, y))
            reward = -np.random.rand()  # Example reward based on congestion/cost
            done = len(self.placed_blocks) >= self.num_blocks
        
        return self.grid, reward, done, {}

    def render(self, mode='human'):
        print(self.grid)
