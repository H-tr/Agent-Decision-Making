def generate_config(maze_size: int, save_path: str):
    import yaml
    config = {
        "size": 4,
        "walls": [[1, 1]],
        "terminal_states": [[0, 0], [3, 3]],
        "rewards": {
            "[0, 0]": 1,
            "[3, 3]": -1
        },
        "transition_prob": 0.8,
        "discount": 0.9,
        "white_reward": -0.04
    }
    with open("config/default.yaml", "w") as file:
        yaml.dump(config, file)