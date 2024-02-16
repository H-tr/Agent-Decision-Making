

class Gridworld:
    def __init__(
        self, size, walls, terminal_states, rewards, transition_prob=1.0, discount=0.9
    ):
        self.size = size
        self.walls = walls
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.transition_prob = transition_prob
        self.discount = discount
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_next_state(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        if (
            next_state in self.walls
            or next_state[0] < 0
            or next_state[0] >= self.size[0]
            or next_state[1] < 0
            or next_state[1] >= self.size[1]
        ):
            next_state = state
        return next_state

    def get_reward(self, state, action, next_state):
        return self.rewards.get(next_state, 0)
