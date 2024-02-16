import numpy as np


def policy_iteration(env, threshold=0.001):
    # Initialize policy as an empty array with dtype=object
    policy = np.empty(env.size, dtype=object)
    # Fill the policy array with the default action (0, 1)
    for i in range(env.size[0]):
        for j in range(env.size[1]):
            policy[i, j] = (0, 1)
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
