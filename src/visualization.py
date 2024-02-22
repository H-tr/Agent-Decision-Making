from typing import Dict, List
import matplotlib.pyplot as plt
from src.gridworld import Gridworld
import numpy as np


class Visualizer:
    def __init__(self, env: Gridworld) -> None:
        self.size = env.size
        self.walls = env.walls
        self.rewards = env.rewards

        # Initialize a matrix to represent the grid for coloring
        self.grid_color = np.ones((self.size[0], self.size[1], 3))  # White by default

        # Define colors
        self.wall_color = np.array([0.5, 0.5, 0.5])  # Gray for walls
        self.positive_reward_color = np.array([0, 1, 0])  # Green for positive rewards
        self.negative_reward_color = np.array(
            [1, 0.5, 0]
        )  # Orange for negative rewards

        for y in range(self.size[0]):
            for x in range(self.size[1]):
                if (y, x) in self.walls:
                    self.grid_color[y, x] = self.wall_color
                elif (y, x) in self.rewards and self.rewards[(y, x)] > 0:
                    self.grid_color[y, x] = self.positive_reward_color
                elif (y, x) in self.rewards and self.rewards[(y, x)] < 0:
                    self.grid_color[y, x] = self.negative_reward_color

    def get_fig(self):
        fig, ax = plt.subplots()
        # Apply colors based on walls and rewards
        # Draw grid lines
        ax.set_xticks(np.arange(-0.5, self.size[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size[0], 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Remove axis labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def visualize_board(self):
        ax = self.get_fig()
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                if (y, x) in self.walls:
                    continue
                if (y, x) in self.rewards:
                    # Write the reward value, color is black, bold
                    ax.text(
                        x,
                        y,
                        str(self.rewards[(y, x)]),
                        color="black",
                        ha="center",
                        va="center",
                        weight="bold",
                    )
        # Display the grid with colors
        ax.imshow(self.grid_color, interpolation="nearest")

    # Visualize the utility values
    def visualize_utilities(self, utilities):
        ax = self.get_fig()
        # Draw utility values
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                if (y, x) in self.walls:
                    continue
                if (y, x) in self.rewards:
                    # Write the reward value, color is black, bold
                    ax.text(
                        x,
                        y,
                        str(self.rewards[(y, x)]),
                        color="black",
                        ha="center",
                        va="center",
                        weight="bold",
                    )
                    continue
                ax.text(
                    x,
                    y,
                    f"{utilities[y, x]:.2f}",
                    color="black",
                    ha="center",
                    va="center",
                )

        ax.imshow(self.grid_color, interpolation="nearest")
        plt.show()

    def visualize_policy(self, policy):
        ax = self.get_fig()

        # Draw policy arrows
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                if (y, x) in self.walls:
                    continue  # Skip walls
                if (y, x) in self.rewards:
                    # Write the reward value, color is black, bold
                    ax.text(
                        x,
                        y,
                        str(self.rewards[(y, x)]),
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

        ax.imshow(self.grid_color, interpolation="nearest")
        plt.show()


def display_convergence(log_list: List[List[Dict[int, float]]], name_list: List[str]):
    """Take arbitrary number of logs and visualize the convergence of the algorithm"""
    fig, ax = plt.subplots()
    for i, log in enumerate(log_list):
        ax.plot([list(d.values())[0] for d in log], label=name_list[i])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Utility Value")
    ax.set_yscale("linear")
    ax.legend()
    plt.show()
