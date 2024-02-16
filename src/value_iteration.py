import numpy as np
from .gridworld import Gridworld


def value_iteration(env: Gridworld, threshold: float = 0.001) -> np.ndarray:
    """
    Perform value iteration to find the optimal utilities.

    Parameters:
    - env: Gridworld - The gridworld environment.
    - threshold: float (default 0.001) - The threshold for convergence.

    Returns:
    - np.ndarray - The utilities of all states after convergence.
    """
    V = np.zeros(env.size)
    while True:
        delta = 0
        for i in range(env.size[0]):
            for j in range(env.size[1]):
                if (i, j) in env.walls or (i, j) in env.terminal_states:
                    continue
                v = V[i, j]
                V[i, j] = max(
                    [
                        env.get_reward((i, j), a, env.get_next_state((i, j), a))
                        + env.discount * V[env.get_next_state((i, j), a)]
                        for a in env.actions
                    ]
                )
                delta = max(delta, abs(v - V[i, j]))
        if delta < threshold:
            break
    return V
