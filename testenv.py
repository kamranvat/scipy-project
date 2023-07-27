"""
Just for testing purposes.
Run a gym environment with a stable_baselines3 policy.
Later, test logging the data.
"""

import gymnasium as gym
#import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import time

from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode = 'human')
#obs_space = env.observation_space
#act_space = env.action_space
#obs = env.reset()

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")