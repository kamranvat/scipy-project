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
import os 
import json

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

agent_list = [
    {"name": "A2C", "description": "", "active": True},
    {"name": "ACER", "description": "", "active": False},
    {"name": "ACKTR", "description": "", "active": False},
    {"name": "DQN", "description": "", "active": True},
    {"name": "GAIL", "description": "", "active": False},
    {"name": "PPO1", "description": "", "active": False},
    {"name": "PPO2", "description": "", "active": True},
    {"name": "TRPO", "description": "", "active": False},
]

settings_folder = "./settings"
settings_file = "agent_settings.json"
settings_path = os.path.join(settings_folder, settings_file)


def toggle_active_agents():
    """Let the user toggle which agents should be included for training via CLI"""
    while True:
        print("\n Current Policies: ")
        print_active_agents(agent_list)

        choice = input(
            "\n Enter the name of the policy to toggle (type 'exit' to quit): "
        )

        if choice.lower() == "exit":
            break

        found = False
        for agent in agent_list:
            if agent["name"] == choice:
                agent["active"] = not agent["active"]
                found = True
                print(
                    f"Policy '{agent['name']}' is now {'Active' if agent['active'] else 'Inactive'}.\n"
                )
                break

        if not found:
            print(
                f"Policy with name '{choice}' not found. Please enter a valid name or 'exit' to quit."
            )


def print_active_agents(agent_list):
    """Format and print the agent list, sorted by active/inactive"""
    active_names = [
        agent["name"] for agent in agent_list if agent.get("active") == True
    ]
    inactive_names = [
        agent["name"] for agent in agent_list if agent.get("active") == False
    ]

    print("Active: \n".join(active_names))
    print("\n Inactive: \n".join(inactive_names))


def save_settings(agent_list):
    if not os.path.exists(settings_folder):
        os.makedirs(settings_folder)

    with open(settings_path, "w") as file:
        json.dump(agent_list, file, indent=4)

def load_settings():
    if os.path.exists(settings_path):
        with open(settings_path, "r") as file:
            return json.load(file)
    else:
        return []


def compare():
    # call the comparison script with the right values
    pass


def compare_shipped():
    # like compare but with our trained values(optional)
    pass


def display_settings():
    # print the current algorithm list
    pass
