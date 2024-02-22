# Maze Solver: Intelligent Agents Assignment 1

[![Python Tests](https://github.com/H-tr/Agent-Decision-Making/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/H-tr/Agent-Decision-Making/actions/workflows/python-test.yml)

![logo](asset/logo.webp)

## Overview

This project is the first assignment for the SC4003-CE4046-CZ4046-INTELLIGENT AGENTS course. It involves implementing value iteration and policy iteration algorithms to solve a maze environment. The goal is to find the optimal policy and utilities for all non-wall states in the maze.

## Results

The following section presents the results of the maze solver algorithm, showcasing the original maze configuration and the optimized policy obtained after running the algorithm.

### Original Maze

The original maze is set up with walls, positive rewards, and negative rewards as shown below:

![Original Maze](asset/ori_maze.png)

### Optimized Policy

After running the maze solver algorithm, the optimized policy indicating the best actions at each state is visualized below:

Value iteration policy and utility:

Policy                     |  Utility
:-------------------------:|:-------------------------:
![Value Iteration Policy](asset/value_iteration_policy.png)  |  ![Value Iteration Utility](asset/value_iteration_utility.png)

Policy iteration policy and utility:

Policy                     |  Utility
:-------------------------:|:-------------------------:
![Policy Iteration Policy](asset/policy_iteration_policy.png)  |  ![Policy Iteration Utility](asset/policy_iteration_utility.png)

The arrows represent the direction of the optimal action to take from each non-wall grid cell. Green cells indicate positive rewards, orange cells indicate negative rewards, and gray cells represent walls. The optimized policy provides a guide for an agent to maximize rewards and reach the goal state efficiently.

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/H-tr/Agent-Decision-Making.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Agent-Decision-Making
   ```

## Usage

1. Create the python environment

```bash
conda create -n maze_solver python=3.10 -y
conda activate maze_solver
pip install -r requirements.txt
```

1. Run the main logic

To monitor the iteration progress

```bash
tensorboard --logdir=runs
```

```bash
python main.py
```

1. Test

```bash
python -m unittest tests.test_maze_solver
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Course instructors and teaching assistants for providing guidance and support.
