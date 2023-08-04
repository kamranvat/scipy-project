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
import training
import cli

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
    help="amount of steps each agent should be trained for (default: 5,000)",
)

args = parser.parse_args()

default_agent_list = [
    {
        "name": "A2C",
        "active": True,
        "description": "A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C). It uses multiple workers to avoid the use of a replay buffer.",
    },
    {
        "name": "DQN",
        "active": True,
        "description": "Deep Q Network (DQN) builds on Fitted Q-Iteration (FQI) and make use of different tricks to stabilize the learning with neural networks: it uses a replay buffer, a target network and gradient clipping.",
    },
    {
        "name": "PPO",
        "active": True,
        "description": "The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor).",
    },
]


def compare():
    # call the comparison script with the right values
    pass


def compare_shipped():
    # like compare but with our trained values(optional)
    pass