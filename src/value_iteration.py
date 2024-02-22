import numpy as np
from .gridworld import Gridworld
from tensorboardX import SummaryWriter


def get_policy(env: Gridworld, utilities: np.ndarray) -> np.ndarray:
    """
    Get the optimal policy based on the utilities.

    Parameters:
    - env: Gridworld - The gridworld environment.
    - utilities: np.ndarray - The utilities of all states.

    Returns:
    - np.ndarray - The optimal policy.
    """
    policy = np.zeros(env.size, dtype=object)
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    for i in range(env.size[0]):
        for j in range(env.size[1]):
            if (
                (i, j) in env.walls
                or (i, j) in env.terminal_states
                or (i, j) in env.rewards.keys()
            ):
                continue
            q_values = [
                sum(
                    prob
                    * (
                        env.get_reward((i, j), action, next_state)
                        + env.discount * utilities[next_state]
                    )
                    for next_state, prob in env.get_transition_states_and_probs(
                        (i, j), action
                    )
                )
                for action in env.actions
            ]
            policy[i, j] = actions[np.argmax(q_values)]
    return policy


def value_iteration(
    env: Gridworld, threshold: float = 0.001, min_iteration: int = 50
) -> np.ndarray:
    """
    Perform value iteration to find the optimal utilities.

    Parameters:
    - env: Gridworld - The gridworld environment.
    - threshold: float (default 0.001) - The threshold for convergence.

    Returns:
    - np.ndarray - The utilities of all states after convergence.
    """
    writer = SummaryWriter("runs/maze_solver_experiment")
    V = np.zeros(env.size)
    iteration = 0
    while True:
        delta = 0
        for i in range(env.size[0]):
            for j in range(env.size[1]):
                if (
                    (i, j) in env.walls
                    or (i, j) in env.terminal_states
                    or (i, j) in env.rewards.keys()
                ):
                    continue
                v = V[i, j]
                V[i, j] = max(
                    [
                        sum(
                            prob
                            * (
                                env.get_reward((i, j), action, next_state)
                                + env.discount * V[next_state]
                            )
                            for next_state, prob in env.get_transition_states_and_probs(
                                (i, j), action
                            )
                        )
                        for action in env.actions
                    ]
                )
                delta = max(delta, abs(v - V[i, j]))

        writer.add_scalar("Value Iteration Delta", delta, iteration)
        writer.add_scalar("Value Iteration Utilities", V.mean(), iteration)
        iteration += 1
        if delta < threshold and iteration > min_iteration:
            break
    return V
