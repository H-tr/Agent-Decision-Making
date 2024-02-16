import numpy as np
from .gridworld import Gridworld
from typing import Tuple


def policy_iteration(
    env: Gridworld, threshold: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform policy iteration to find the optimal policy and utilities.

    Parameters:
    - env: Gridworld - The gridworld environment.
    - threshold: float (default 0.001) - The threshold for convergence.

    Returns:
    - Tuple[np.ndarray, np.ndarray] - A tuple containing the optimal policy and the utilities of all states.
    """
    policy = np.empty(env.size, dtype=object)
    for i in range(env.size[0]):
        for j in range(env.size[1]):
            policy[i, j] = (0, 1)  # Initialize with a default action
    V = np.zeros(env.size)
    while True:
        # Policy evaluation
        while True:
            delta = 0
            for i in range(env.size[0]):
                for j in range(env.size[1]):
                    if (i, j) in env.walls or (i, j) in env.terminal_states:
                        continue
                    v = V[i, j]
                    V[i, j] = (
                        env.get_reward(
                            (i, j),
                            policy[i, j],
                            env.get_next_state((i, j), policy[i, j]),
                        )
                        + env.discount * V[env.get_next_state((i, j), policy[i, j])]
                    )
                    delta = max(delta, abs(v - V[i, j]))
            if delta < threshold:
                break

        # Policy improvement
        policy_stable = True
        for i in range(env.size[0]):
            for j in range(env.size[1]):
                if (i, j) in env.walls or (i, j) in env.terminal_states:
                    continue
                old_action = policy[i, j]
                policy[i, j] = max(
                    env.actions,
                    key=lambda a: env.get_reward(
                        (i, j), a, env.get_next_state((i, j), a)
                    )
                    + env.discount * V[env.get_next_state((i, j), a)],
                )
                if old_action != policy[i, j]:
                    policy_stable = False
        if policy_stable:
            break
    return policy, V
