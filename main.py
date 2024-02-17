from src.gridworld import Gridworld
from src.value_iteration import value_iteration
from src.policy_iteration import policy_iteration
from src.visualization import visualize_gridworld
import yaml


def main():
    # Load the configuration from the YAML file
    with open("config/default.yaml", "r") as file:
        config = yaml.safe_load(file)
    size = config["size"]
    walls = config["walls"]
    terminal_states = config["terminal_states"]
    rewards = {
        tuple(map(int, key.strip("[]").split(","))): value
        for key, value in config["rewards"].items()
    }

    env = Gridworld(
        size,
        walls,
        terminal_states,
        rewards,
        config["transition_prob"],
        config["discount"],
        config["white_reward"],
    )

    V_value_iter = value_iteration(env)
    print("Value Iteration:")
    print(V_value_iter)

    policy, V_policy_iter = policy_iteration(env)
    print("Policy Iteration:")
    print("Policy:", policy)
    print("Utilities:", V_policy_iter)

    # Visualize the gridworld with the optimal policy
    visualize_gridworld(size, walls, rewards, policy)


if __name__ == "__main__":
    main()
