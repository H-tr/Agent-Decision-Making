import numpy as np


def value_iteration(env, threshold=0.001):
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
