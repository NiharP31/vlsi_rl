import numpy as np

def augment_data(grid):
    # Augmentation Techniques:
    # 1. Flipping
    augmented_grid = np.flipud(grid) if np.random.rand() > 0.5 else grid
    augmented_grid = np.fliplr(augmented_grid) if np.random.rand() > 0.5 else augmented_grid

    # 2. Rotation
    rotations = np.random.choice([0, 1, 2, 3])  # 0 to 3 rotations of 90 degrees
    augmented_grid = np.rot90(augmented_grid, k=rotations)

    # 3. Adding Noise
    noise = np.random.normal(0, 0.1, augmented_grid.shape)  # Add some Gaussian noise
    augmented_grid = np.clip(augmented_grid + noise, 0, 1)  # Ensure values remain between 0 and 1

    return augmented_grid
