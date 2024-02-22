from rich.console import Console
import random

CONSOLE = Console()


def generate_config(maze_size: int, save_path: str):
    import yaml

    config = {
        "size": [maze_size, maze_size],
        "terminal_states": [],
        "rewards": {},
        "transition_prob": 0.8,
        "discount": 0.99,
        "white_reward": -0.04,
    }
    # Randomly generate maze_size - 1 number of walls, no repeat
    walls = []
    while len(walls) < maze_size - 1:
        wall = [random.randint(0, maze_size - 1), random.randint(0, maze_size - 1)]
        if wall not in walls:
            walls.append(wall)
    config["walls"] = walls

    # Randomly generate 1/3 of maze_size ** 2 number of reward states, no repeat
    reward_states = []
    while len(reward_states) < maze_size**2 // 3:
        reward = [random.randint(0, maze_size - 1), random.randint(0, maze_size - 1)]
        if reward not in reward_states:
            reward_states.append(reward)

    for reward in reward_states:
        config["rewards"][f"{reward}"] = random.choice([1, -1])

    with open(save_path, "w") as file:
        yaml.dump(config, file)
