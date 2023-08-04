"""Contains everything required to train models after being passed a list of dicts with model names"""

# TODO  functions to pull the names out of the dicts, call the right algos

import gymnasium as gym
#import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from stable_baselines3.common.logger import configure

# TODO double check the imported algos here on the website of sd3
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO



def call_active_models(agent_list):
    """
    Takes a list of dictionaries that include a "name" key and value. 
    Calls training for each model that is named in the list.
    """
    for agent in agent_list:
        if agent.get("active") == True:
            policy_name = agent.get("name")
            #environment_name = agent.get("environment")
            environment_name = "CartPole-v1"
            model_args = agent.get("model_args", {})

            # "Translate" from str to class
            policy_classes = {
                "A2C": A2C,
                "DQN": DQN,
                "PPO": PPO,
            }

            if policy_name in policy_classes:
                policy_class = policy_classes[policy_name]
                model = policy_class(**model_args, env=environment_name, verbose=1)

                # Call the policy here or use it as needed 
                # TODO potentially add something here idk
                print(f"Policy {policy_name} is active. Model: {model}")
            else:
                print(f"Policy {policy_name} is active, but the class is not defined.")

    print("\n" * 3)
