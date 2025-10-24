import argparse
from agent import agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, required=True)
    args = parser.parse_args()
    print(agent(args.observation))
