import os
import json

settings_folder = "./settings"
settings_file = "agent_settings.json"
settings_path = os.path.join(settings_folder, settings_file)

default_agent_list = [
    {"name": "A2C", "description": "", "active": True},
    {"name": "ACER", "description": "", "active": False},
    {"name": "ACKTR", "description": "", "active": False},
    {"name": "DQN", "description": "", "active": True},
    {"name": "GAIL", "description": "", "active": False},
    {"name": "PPO1", "description": "", "active": False},
    {"name": "PPO2", "description": "", "active": True},
    {"name": "TRPO", "description": "", "active": False},
]


def toggle_active_agents():
    """Let the user toggle which agents should be included for training via CLI"""
    while True:
        if settings_file_exists():
            agent_list = load_settings()
        else:
            agent_list = default_agent_list
            save_settings(agent_list)

        print("\n Current Policies: ")
        print_agents_by_activity(agent_list)

        choice = input(
            "\n Enter the name of the policy to toggle (type 'exit' to quit): "
        )

        if choice.lower() == "exit":
            save_settings(agent_list)
            break

        found = False
        for agent in agent_list:
            if agent["name"] == choice:
                agent["active"] = not agent["active"]
                found = True
                print(
                    f"Policy '{agent['name']}' is now {'Active' if agent['active'] else 'Inactive'}.\n"
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

    print("Active: \n".join(active_names))
    print("\n Inactive: \n".join(inactive_names))


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
