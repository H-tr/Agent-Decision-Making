import argparse
from src.core import solve_maze


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--part", type=int, default="1")
    args = argparser.parse_args()
    
    if args.part == 1:
        solve_maze("config/default.yaml")

if __name__ == "__main__":
    main()
