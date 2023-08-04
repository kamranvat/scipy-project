import os
import json

settings_folder = "./settings"
settings_file = "agent_settings.json"
settings_path = os.path.join(settings_folder, settings_file)

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


def toggle_active_agents():
    """Let the user toggle which agents should be included for training via CLI"""
    while True:

        # Default settings if no file exists
        if settings_file_exists():
            agent_list = load_settings()
        else:
            agent_list = default_agent_list
            save_settings(agent_list)

        print("\nCurrent Policies: ")
        print_agents_by_activity(agent_list)

        choice = input(
            "\nEnter the name of the policy to toggle (type 'exit' to quit): "
        ).upper()

        if choice == "EXIT":
            save_settings(agent_list)
            break

        found = False
        for agent in agent_list:
            if agent["name"] == choice:
                agent["active"] = not agent["active"]
                found = True
                print(
                    f"Policy '{agent['name']}' set to {'Active' if agent['active'] else 'Inactive'}. "
                )
                save_settings(agent_list)
                break

        if not found:
            print(
                f"Policy with name '{choice}' not found. Please enter a valid name or 'exit' to quit."
            )


def print_agents_by_activity(agent_list):
    """Format and print the agent list, sorted by active/inactive"""
    active_names = [
        agent["name"] for agent in agent_list if agent.get("active") == True
    ]
    inactive_names = [
        agent["name"] for agent in agent_list if agent.get("active") == False
    ]

    print("\n Active: \n -" + "\n -".join(active_names))
    print("\n Inactive: \n -" + "\n -".join(inactive_names))


def save_settings(agent_list):
    """Save the agent_list in json format at settings path"""
    if not os.path.exists(settings_folder):
        os.makedirs(settings_folder)

    with open(settings_path, "w") as file:
        json.dump(agent_list, file, indent=4)


def load_settings():
    """Load and return a json file from the settings path"""
    if os.path.exists(settings_path):
        with open(settings_path, "r") as file:
            return json.load(file)
    else:
        raise FileNotFoundError(
            f"The settings file '{settings_file}' does not exist in the '{settings_folder}' folder."
        )


def settings_file_exists():
    """Returns True if agent_settings.json exists"""
    settings_folder = "./settings"
    settings_file = "agent_settings.json"
    settings_path = os.path.join(settings_folder, settings_file)

    return os.path.exists(settings_path)
