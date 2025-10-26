import argparse
from qwafia import revac
agent=revac()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, required=True)
    args = parser.parse_args()
    print(agent(args.observation))
