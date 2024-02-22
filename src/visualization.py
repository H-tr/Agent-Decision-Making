import matplotlib.pyplot as plt
import numpy as np


# Visualize the utility values
def visualize_utilities(size, walls, rewards, utilities):
    fig, ax = plt.subplots()
    # Initialize a matrix to represent the grid for coloring
    grid_color = np.ones((size[0], size[1], 3))  # White by default

    # Define colors
    wall_color = np.array([0.5, 0.5, 0.5])  # Gray for walls
    positive_reward_color = np.array([0, 1, 0])  # Green for positive rewards
    negative_reward_color = np.array([1, 0.5, 0])  # Orange for negative rewards

    # Apply colors based on walls and rewards
    for y in range(size[0]):
        for x in range(size[1]):
            if (y, x) in walls:
                grid_color[y, x] = wall_color
            elif (y, x) in rewards and rewards[(y, x)] > 0:
                grid_color[y, x] = positive_reward_color
            elif (y, x) in rewards and rewards[(y, x)] < 0:
                grid_color[y, x] = negative_reward_color

    # Display the grid with colors
    ax.imshow(grid_color, interpolation="nearest")

    # Draw utility values
    for y in range(size[0]):
        for x in range(size[1]):
            if (y, x) in walls:
                continue
            if (y, x) in rewards:
                # Write the reward value, color is black, bold
                ax.text(
                    x,
                    y,
                    str(rewards[(y, x)]),
                    color="black",
                    ha="center",
                    va="center",
                    weight="bold",
                )
                continue
            ax.text(
                x, y, f"{utilities[y, x]:.2f}", color="black", ha="center", va="center"
            )

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, size[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size[0], 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def visualize_gridworld(size, walls, rewards, policy):
    fig, ax = plt.subplots()
    # Initialize a matrix to represent the grid for coloring
    grid_color = np.ones((size[0], size[1], 3))  # White by default

    # Define colors
    wall_color = np.array([0.5, 0.5, 0.5])  # Gray for walls
    positive_reward_color = np.array([0, 1, 0])  # Green for positive rewards
    negative_reward_color = np.array([1, 0.5, 0])  # Orange for negative rewards

    # Apply colors based on walls and rewards
    for y in range(size[0]):
        for x in range(size[1]):
            if (y, x) in walls:
                grid_color[y, x] = wall_color
            elif (y, x) in rewards and rewards[(y, x)] > 0:
                grid_color[y, x] = positive_reward_color
            elif (y, x) in rewards and rewards[(y, x)] < 0:
                grid_color[y, x] = negative_reward_color

    # Display the grid with colors
    ax.imshow(grid_color, interpolation="nearest")

    # Draw policy arrows
    for y in range(size[0]):
        for x in range(size[1]):
            if (y, x) in walls:
                continue  # Skip walls
            if (y, x) in rewards:
                # Write the reward value, color is black, bold
                ax.text(
                    x,
                    y,
                    str(rewards[(y, x)]),
                    color="black",
                    ha="center",
                    va="center",
                    weight="bold",
                )
                continue
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

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, size[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size[0], 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
