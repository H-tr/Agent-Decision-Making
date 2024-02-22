import unittest
from src.gridworld import Gridworld
from src.value_iteration import value_iteration
from src.policy_iteration import policy_iteration


class TestMazeSolver(unittest.TestCase):
    def setUp(self):
        self.size = (4, 4)
        self.walls = [(1, 1), (1, 3), (2, 1), (2, 3)]
        self.terminal_states = [(0, 0), (3, 3)]
        self.rewards = {(3, 3): 1, (0, 0): -1}
        self.env = Gridworld(self.size, self.walls, self.terminal_states, self.rewards)

    def test_value_iteration(self):
        V, _ = value_iteration(self.env)
        self.assertEqual(V.shape, self.size)
        # Add more assertions to check the correctness of the utilities

    def test_policy_iteration(self):
        policy, V, _ = policy_iteration(self.env)
        self.assertEqual(policy.shape, self.size)
        self.assertEqual(V.shape, self.size)
        # Add more assertions to check the correctness of the policy and utilities


if __name__ == "__main__":
    unittest.main()
