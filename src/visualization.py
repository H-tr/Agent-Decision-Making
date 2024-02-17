import matplotlib.pyplot as plt
import numpy as np


def visualize_gridworld(size, walls, rewards, policy):
    fig, ax = plt.subplots()
    # Initialize a matrix to represent the grid for coloring
    grid_color = np.ones((size[0], size[1], 3))  # White by default

    # Colors
    wall_color = np.array([0.5, 0.5, 0.5])  # Gray
    positive_reward_color = np.array([0, 1, 0])  # Green
    negative_reward_color = np.array([1, 0.5, 0])  # Orange

    # Apply colors
    for y in range(size[0]):
        for x in range(size[1]):
            if (y, x) in walls:
                grid_color[y, x] = wall_color
            elif (y, x) in rewards:
                reward = rewards[(y, x)]
                if reward > 0:
                    grid_color[y, x] = positive_reward_color
                elif reward < 0:
                    grid_color[y, x] = negative_reward_color

    # Create a color map image
    ax.imshow(grid_color, interpolation="nearest")

    # Draw policy arrows
    for y in range(size[0]):
        for x in range(size[1]):
            if (y, x) in walls:
                continue  # Skip walls and terminal states
            action = policy[y, x]
            dx, dy = 0, 0
            if action == (0, 1):  # Right
                dx, dy = 0.4, 0
            elif action == (1, 0):  # Down
                dx, dy = 0, 0.4
            elif action == (0, -1):  # Left
                dx, dy = -0.4, 0
            elif action == (-1, 0):  # Up
                dx, dy = 0, -0.4
            ax.arrow(
                x - dx / 2,
                y - dy / 2,
                dx,
                dy,
                head_width=0.1,
                head_length=0.1,
                fc="k",
                ec="k",
            )

    # Remove axis markings for clarity
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
