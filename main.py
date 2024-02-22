from src.gridworld import Gridworld
from src.value_iteration import value_iteration, get_policy
from src.policy_iteration import policy_iteration
from src.visualization import Visualizer
import yaml


def main():
    # Load the configuration from the YAML file
    with open("config/default.yaml", "r") as file:
        config = yaml.safe_load(file)
    size = config["size"]
    walls = config["walls"]
    # Change list into tuple
    walls = [tuple(wall) for wall in walls]
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

    visualizer = Visualizer(env=env)

    # Value iteration
    V_value_iter = value_iteration(env)
    print("Value Iteration:")
    print(V_value_iter)

    # Get the optimal policy
    policy = get_policy(env, V_value_iter)
    print("Policy:", policy)

    # Visualize the gridworld with the optimal policy
    visualizer.visualize_board()
    visualizer.visualize_policy(policy)
    visualizer.visualize_utilities(V_value_iter)

    # Policy iteration
    policy, V_policy_iter = policy_iteration(env)
    print("Policy Iteration:")
    print("Policy:", policy)
    print("Utilities:", V_policy_iter)

    # Visualize the gridworld with the optimal policy
    visualizer.visualize_policy(policy)
    visualizer.visualize_utilities(V_policy_iter)


if __name__ == "__main__":
    main()
