from src.gridworld import Gridworld
from src.value_iteration import value_iteration
from src.policy_iteration import policy_iteration


def main():
    size = (6, 6)
    walls = [(0, 1), (1, 4), (4, 1), (4, 2), (4, 3)]
    terminal_states = []
    rewards = {(0, 0): 1, (0, 2): 1, (0, 5): 1, (1, 3): 1, (2, 4): 1, (3, 5):1,
               (1, 1): -1, (1, 5): -1, (2, 2): -1, (3, 3): -1, (4, 4): -1}
    env = Gridworld(size, walls, terminal_states, rewards)

    V_value_iter = value_iteration(env)
    print("Value Iteration:")
    print(V_value_iter)

    policy, V_policy_iter = policy_iteration(env)
    print("Policy Iteration:")
    print("Policy:", policy)
    print("Utilities:", V_policy_iter)


if __name__ == "__main__":
    main()
