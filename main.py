import argparse
from src.core import solve_maze
from src.utils import generate_config
from src.visualization import display_convergence


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--assignment",
        type=str,
        default="part_1",
        choices=["part_1", "part_2"],
        help="The assignment to run.",
    )
    argparser.add_argument(
        "--no_of_solve",
        type=int,
        default=6,
        help="How many times to solve the increased size maze.",
    )
    argparser.add_argument(
        "--increase_interval",
        type=int,
        default=2,
        help="How much to increase the size of the maze each generation.",
    )
    args = argparser.parse_args()

    if args.assignment == "part_1":
        solve_maze("config/default.yaml")
    elif args.assignment == "part_2":
        value_logs = {}
        policy_logs = {}
        for i in range(args.no_of_solve):
            maze_size = i * args.increase_interval + 7
            generate_config(maze_size, f"config/maze_{maze_size}.yaml")
            value_log, policy_log = solve_maze(f"config/maze_{maze_size}.yaml")
            value_logs[f"value_iter_size_{maze_size}"] = value_log
            policy_logs[f"policy_iter_size_{maze_size}"] = policy_log
        # Add code to visualize the convergence of the value and policy iteration
        display_convergence(list(value_logs.values()), list(value_logs.keys()))
        display_convergence(list(policy_logs.values()), list(policy_logs.keys()))
    else:
        raise ValueError("Invalid assignment")


if __name__ == "__main__":
    main()
