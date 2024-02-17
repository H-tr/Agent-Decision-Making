from typing import List, Tuple, Dict


class Gridworld:
    def __init__(
        self,
        size: Tuple[int, int],
        walls: List[Tuple[int, int]],
        terminal_states: List[Tuple[int, int]],
        rewards: Dict[Tuple[int, int], float],
        transition_prob: float = 0.8,
        discount: float = 0.99,
        white_reward: float = -0.04,
    ):
        """
        Initialize the Gridworld environment.

        Parameters:
        - size: Tuple[int, int] - The dimensions of the gridworld (rows, columns).
        - walls: List[Tuple[int, int]] - A list of coordinates representing wall positions.
        - terminal_states: List[Tuple[int, int]] - A list of coordinates representing terminal states.
        - rewards: Dict[Tuple[int, int], float] - A dictionary mapping state coordinates to rewards.
        - transition_prob: float (default 0.8) - The probability of successfully moving in the intended direction.
        - discount: float (default 0.99) - The discount factor for future rewards.
        """
        self.size = size
        self.walls = walls
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.transition_prob = transition_prob
        self.discount = discount
        self.white_reward = white_reward
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """
        Check if a state is terminal.

        Parameters:
        - state: Tuple[int, int] - The coordinates of the state to check.

        Returns:
        - bool - True if the state is terminal, False otherwise.
        """
        return state in self.terminal_states

    def rotate_action(self, action: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """
        Rotate an action 90 degrees left or right.

        Parameters:
        - action: Tuple[int, int] - The original action (dx, dy).
        - direction: int - The direction to rotate the action. -1 for left, 1 for right.

        Returns:
        - Tuple[int, int] - The rotated action.
        """
        # Mapping of actions to their left/right rotations
        # Action format is (dx, dy)
        action_rotations = {
            (0, 1): {  # Right
                -1: (-1, 0),  # Rotate left to Up
                1: (1, 0),  # Rotate right to Down
            },
            (1, 0): {  # Down
                -1: (0, 1),  # Rotate left to Right
                1: (0, -1),  # Rotate right to Left
            },
            (0, -1): {  # Left
                -1: (1, 0),  # Rotate left to Down
                1: (-1, 0),  # Rotate right to Up
            },
            (-1, 0): {  # Up
                -1: (0, -1),  # Rotate left to Left
                1: (0, 1),  # Rotate right to Right
            },
        }

        return action_rotations[action][direction]

    def get_transition_states_and_probs(
        self, state: Tuple[int, int], action: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Given a state and an action, return a list of (next_state, probability) pairs."""
        if state in self.terminal_states:
            return [(state, 1.0)]

        forward = self.get_next_state(state, action)
        left = self.get_next_state(state, self.rotate_action(action, -1))
        right = self.get_next_state(state, self.rotate_action(action, 1))

        # Staying in the current state if all moves lead to walls or out of bounds
        if forward == state and left == state and right == state:
            return [(state, 1.0)]

        transitions = [
            (forward, self.transition_prob),
            (left, (1 - self.transition_prob) / 2),
            (right, (1 - self.transition_prob) / 2),
        ]

        # Removing duplicates if any (e.g., when left and right result in the same state)
        results = {}
        for next_state, prob in transitions:
            if next_state in results:
                results[next_state] += prob
            else:
                results[next_state] = prob

        return list(results.items())

    def get_next_state(
        self, state: Tuple[int, int], action: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get the next state given a state and action.

        Parameters:
        - state: Tuple[int, int] - The current state coordinates.
        - action: Tuple[int, int] - The action to be taken (delta row, delta column).

        Returns:
        - Tuple[int, int] - The coordinates of the next state.
        """
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

    def get_reward(
        self,
        state: Tuple[int, int],
        action: Tuple[int, int],
        next_state: Tuple[int, int],
    ) -> float:
        """
        Get the reward for a state transition.

        Parameters:
        - state: Tuple[int, int] - The current state coordinates.
        - action: Tuple[int, int] - The action taken.
        - next_state: Tuple[int, int] - The resulting state coordinates.

        Returns:
        - float - The reward for the transition.
        """
        return self.rewards.get(next_state, self.white_reward)
