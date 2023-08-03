"""
Just for testing purposes.
Run a gym environment with a stable_baselines3 policy.
Log the data to csv
"""

import gymnasium as gym
#import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.logger import CSVOutputFormat as csvformat

# Configure logging
tmp_path = "./logs/"
new_logger = configure(tmp_path, ["stdout", "csv"])


# Set model and train
model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
model.set_logger(new_logger)
model.learn(10000)


print("Run environment and render: ")

env = gym.make("CartPole-v1", render_mode = 'human')
vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")