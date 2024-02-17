import numpy as np
from .gridworld import Gridworld
from typing import Tuple
from tensorboardX import SummaryWriter


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
    writer = SummaryWriter("runs/maze_solver_experiment")
    policy = np.empty(env.size, dtype=object)
    for i in range(env.size[0]):
        for j in range(env.size[1]):
            policy[i, j] = (0, 1)  # Initialize with a default action
    V = np.zeros(env.size)
    iteration = 0
    while True:
        # Policy evaluation
        while True:
            delta = 0
            for i in range(env.size[0]):
                for j in range(env.size[1]):
                    if (i, j) in env.walls or (i, j) in env.terminal_states:
                        continue
                    v = V[i, j]
                    action = policy[i, j]
                    V[i, j] = sum(
                        prob
                        * (
                            env.get_reward((i, j), action, next_state)
                            + env.discount * V[next_state]
                        )
                        for next_state, prob in env.get_transition_states_and_probs(
                            (i, j), action
                        )
                    )
                    delta = max(delta, abs(v - V[i, j]))

            writer.add_scalar("Policy Iteration Delta", delta, iteration)
            if delta < threshold:
                break

        # Policy Improvement
        policy_stable = True
        for i in range(env.size[0]):
            for j in range(env.size[1]):
                if (i, j) in env.terminal_states:
                    continue
                old_action = policy[i, j]
                action_values = []
                for action in env.actions:
                    action_value = sum(
                        prob
                        * (
                            env.get_reward((i, j), action, next_state)
                            + env.discount * V[next_state]
                        )
                        for next_state, prob in env.get_transition_states_and_probs(
                            (i, j), action
                        )
                    )
                    action_values.append(action_value)
                best_action = env.actions[np.argmax(action_values)]
                policy[i, j] = best_action
                if old_action != best_action:
                    policy_stable = False

        writer.add_scalar("Policy Iteration Utilities", V.mean(), iteration)
        if policy_stable:
            break
        iteration += 1
    return policy, V
