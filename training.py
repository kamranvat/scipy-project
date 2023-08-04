"""Contains everything required to train models after being passed a list of dicts with model names"""
import os
import gymnasium as gym

from stable_baselines3.common.logger import configure

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO

log_path = "./logs/"


def train_active_models(agent_list, runs):
    """
    Takes a list of dictionaries that include a "name" key and value.
    Calls training for each model that is named in the list.
    """

    # TODO allow user to change number of episodes

    for agent in agent_list:
        if agent.get("active") == True:
            # Configure logging
            csv_logger = configure(log_path, ["stdout", "csv"])

            policy_name = agent.get("name")
            # environment_name = agent.get("environment")
            environment_name = "CartPole-v1"
            # TODO add optimal hyperparameters to model args to get to 6 models
            # model_args = agent.get("model_args", {})

            # "Translate" from str to class
            policy_classes = {
                "A2C": A2C,
                "DQN": DQN,
                "PPO": PPO,
            }

            if policy_name in policy_classes:
                policy_class = policy_classes[policy_name]
                # model = policy_class(**model_args, env=environment_name, verbose=1)
                model = policy_class("MlpPolicy", env=environment_name, verbose=1)

                # Call the policy here or use it as needed
                # TODO potentially add something here idk
                print(
                    f"Policy {policy_name} is active. Model: {model}. Training now..."
                )
                model.set_logger(csv_logger)
                model.learn(runs)
                rename_progress_file(agent.get("name"))

            else:
                print(f"Policy {policy_name} is active, but the class is not defined.")

    print("\n" * 3)


def train_demo():
    """trains A2C, shows the result in render mode 'human'"""
    model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
    model.learn(10000)

    #env = gym.make("CartPole-v1", render_mode = 'human')
    vec_env = model.get_env()
    obs = vec_env.reset()
    episode_reward = 0

    for _ in range(5000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        episode_reward += reward
        
        if done:
            print(episode_reward)
            episode_reward = 0

def rename_progress_file(new_name):
    """rename the csv log file for later processing"""

    # The logger saves as "progress.csv", change this to the current name
    old_file_path = os.path.join(log_path, "progress.csv")
    new_file_path = os.path.join(log_path, f"{new_name}.csv")

    # Check that file exists before renaming
    if os.path.exists(old_file_path):
        os.rename(old_file_path, new_file_path)
        print(f"Log file stored as '{new_name}.csv'")
    else:
        print("Error: 'progress.csv' file not found in the './logs' subfolder.")
