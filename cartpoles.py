"""
Train different policies in the Cartpole v1 environment from stable baselines 3.
Outputs get logged as csv files. 
The results can be visualized and displayed.
"""

# TODO: The user calls this script from their console. The script should provide a few different options:
# train all
# modify list of agents to train
# train none, only visualize the outputs again
# OPTIONAL visualize the outputs we ship with
# OPTIONAL demo mode: train only one, render output
# if no arguments are given, just display current settings

import argparse
from training import train_active_models
from cli import toggle_active_agents

# Define a parser and command line arguments
parser = argparse.ArgumentParser(
    description="Train and compare different policies (with default hyperparameters) in the cartpole-v1 environment from stable-baselines3.",
    epilog="Warning: A high episode count over many models is computationally expensive.",
)

parser.add_argument(
    "--set",
    "-s",
    action="store_true",
    help="view/modify the agent list before training",
)

parser.add_argument(
    "--train",
    "-t",
    action="store_true",
    help="train all agents set to 'active' in the agent list",
)

parser.add_argument(
    "--compare",
    "-c",
    action="store_true",
    help="show only the visualized results from the last run, suppresses --set and --train",
)

parser.add_argument(
    "--demo",
    "-d",
    action="store_true",
    help="demo mode, trains one agent in render mode 'human'",
)

parser.add_argument(
    "--runs",
    "-r",
    type=int,
    default=5000,
    help="amount of steps each agent should be trained for (default: 5,000)",
)

args = parser.parse_args()


def compare():
    # call the comparison script with the right values
    pass


def compare_shipped():
    # like compare but with our trained values(optional)
    pass

if __name__ == "__main__":

    # TODO unify naming scheme (policy/model/agent)
    # TODO use args to decide which functions to call
    toggle_active_agents()
    train_active_models(args.runs)