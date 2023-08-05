import os
import json

settings_folder = "./settings"
settings_file = "agent_settings.json"
settings_path = os.path.join(settings_folder, settings_file)

default_agent_list = [
    {
        "name": "A2C-def",
        "active": False,
        "description": "A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C). It uses multiple workers to avoid the use of a replay buffer.",
    },
    {
        "name": "DQN-def",
        "active": False,
        "description": "Deep Q Network (DQN) builds on Fitted Q-Iteration (FQI) and make use of different tricks to stabilize the learning with neural networks: it uses a replay buffer, a target network and gradient clipping.",
    },
    {
        "name": "DQN-opt",
        "active": False,
        "description": "Deep Q Network (DQN) builds on Fitted Q-Iteration (FQI) and make use of different tricks to stabilize the learning with neural networks: it uses a replay buffer, a target network and gradient clipping.",
        "model_args": {
            "batch_size": 64,
            "buffer_size": 100000,
            "exploration_final_eps": 0.04,
            "exploration_fraction": 0.16,
            "gamma": 0.99,
            "gradient_steps": 128,
            "learning_rate": 0.023,
            "learning_starts": 1000,
            "policy_kwargs": dict(net_arch=[256, 256]),
            "target_update_interval": 10,
            "train_freq": 256
        }
    },
    {
        "name": "PPO-def",
        "active": False,
        "description": "The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor).",
    },
    {
        "name": "PPO-opt",
        "active": False,
        "description": "The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor). Note: the optimized hyperparameters actually call for 100k timesteps",
        "model_args": {
            "batch_size": 32,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "gae_lambda": 0.8,
            "gamma": 0.98,
            "learning_rate": 0.001,
            "n_epochs": 20,
            "n_steps": 256, # 32*8 steps instead of running 8 parallel envs with 32 steps as in the zoo settings
        },
    },
]


def toggle_active_agents():
    """Let the user toggle which agents should be included for training via CLI"""
    while True:
        agent_list = load_settings()

        print("\nCurrent Policies: ")
        print_agents_by_activity(agent_list)

        choice = input(
            "\nEnter the name of the policy to toggle (type 'exit' to quit): "
        )

        if choice.lower() == "exit":
            save_settings(agent_list)
            break

        found = False
        for agent in agent_list:
            if agent["name"].lower() == choice.lower():
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
        # Default settings if no file exists
        print("agent_settings.json not found. Using default settings.")
        save_settings(default_agent_list)
        load_settings()


def settings_file_exists():
    """Returns True if agent_settings.json exists"""
    settings_folder = "./settings"
    settings_file = "agent_settings.json"
    settings_path = os.path.join(settings_folder, settings_file)

    return os.path.exists(settings_path)
