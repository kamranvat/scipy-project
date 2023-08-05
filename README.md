# CartPole-v1 Performance Measurement

We are measuring and showing the performance of different predefined models in the CartPole-v1 Reinforcement Learning environment over a short amount of timesteps.

This project serves as the final project for the Scientific Python course at the University of Osnabrück, summer semester of 2023.


## Table of Contents

- [Goal](#goal)
- [Motivation](#motivation)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Authors](#authors)
- [About RL and the models](#about-reinforcement-learning-and-the-models)
- [About Cartpole](#about-cartpole)

## Goal

The goal of this project is to showcase a cool Reinforcement Learning (RL) demo using stable-baselines3. We aim to create visually appealing graphs that display the performance of different RL policies in the cartpole-v1 environment.
We will log the performance data in a csv format, then showcase the results to allow for conclusions about the performance being made.

## Motivation

Reinforcement Learning is a field of study that enables agents to learn and make decisions in dynamic environments. 
By using stable-baselines3 and cartpole-v1, we can demonstrate how different RL algorithms perform on a classic control problem. 
This allows us to apply our Scientific Python skills to a well-studied, yet interesting group of problems.

## Installation

To get started, follow these steps:

  - Clone this Github repository
  - Navigate to the project directory
  - Install the required packages specified in 'packages.txt' using pip:
    
    `pip install -r packages.txt`

**Warning / known issue:** The training part of the code has been developed and tested using Ubuntu (and it works). 
However, it causes an issue if executed under Windows (being unable to save/load any files). 
With Windows, you can still generate the visualization for the results, as the repository contains .csv files.
Apart from that, a non-Windows device is unfortunately among the requirements.
If, for testing, you need to run the code on a Windows PC, please contact us, and we will figure something out.

## Usage

The core functionality of the project is the ability to train a set of models for an agent in the CartPole-v1 environment, and to display the results.
You can access these functions from your terminal by using `python cartpoles.py` with the appropriate command line arguments, e.g. `python cartpoles.py --help`. 


| Command Line Arguments | Description                                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| `--help`, `-h`        | Shows a help message with an overview akin to this one                                                            |
| `--set`, `-s`         | Opens a command line menu that lets you toggle which models are "active" (will be trained and/or visualized)     |
| `--train`, `-t`       | Trains all models that are set to "active" in the model list                                                     |
| `--compare`, `-c`     | Shows the visualized logs of all models currently set to "active"                                                |
| `--demo`, `-d`        | Demonstrates what the CartPole-v1 environment looks like by training the A2C model, then displaying the resulting agent for a while (rendered in a pygame window). |
| `--runs RUNS`, `-r RUNS` | Amount of timesteps each agent should be trained for (default: 10,000). Can be used in combination with `--demo`.  |



The `--demo` flag overrides `--set`, `--train`, and `--compare`, and just runs the demo mode (for `--runs` timesteps if `--runs` is also specified).

`--set`, `--train`, and `--compare` can be combined, and will be executed in that order.

You can also use each of them on their own, for example, if you want to view your results again without doing another training run.

Please keep in mind that the currently "active" models get visualized; therefore, if you alter them using `--set` and then run `--compare`, you will potentially see outdated graphs in the visualization.

This should not cause any inaccuracies unless you also changed the amount of timesteps between different runs, and then try to visualize those results in the same `--compare` step.


### Model Settings

As explained above, you can choose which models to train. 
There are three models included from stable-baselines3 that can train on the discrete action space / box observation space that this environment provides:
A2C, PPO, and DQN.

You can train each of them with their default hyperparameters by setting **A2C-def**, **PPO-def** or **DQN-def** to active, respectively.

Additionally, we also included those same models, but with optimal hyperparameters taken from [RL-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo/).
You can train these by including **PPO-opt** or **DQN-opt** in your active agents (included by default). There is **no A2C-opt** since the hyperparameters would not differ from **A2C-def**.

If you want to view the respective hyperparameters for yourself, they can be found on RL Baselines3 Zoo's huggingface:
[PPO](https://huggingface.co/sb3/ppo-CartPole-v1), [DQN](https://huggingface.co/sb3/ppo-CartPole-v1), [A2C](https://huggingface.co/sb3/ppo-CartPole-v1)


You can also view their [benchmarks](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md).
Note that all of these values are taken after many more timesteps that our default values here.


### Saving and loading

Whenever you exit the settings menu, your choices get saved into a file in the ./settings folder, so they persist when the program terminates.
The csv logs created when training the models get stored in the ./logs folder, so they also persist and can be visualized when calling cartpoles.py with the compare flag anytime.

Should the save file be deleted or lost, just run cartpoles.py with the --set flag, this should restore the default settings (you may have to run it twice).

Similarly, if the logs should be deleted or lost, you can generate new ones by training the models again.


## Features

- Choose which models should be included in the training and visualization via command line arguments
- Train RL models in the CartPole-v1 environment
- Visualize the reward over timesteps of the active models during training
- Run a demo mode to see an agent in the CartPole-v1 environment in action
- Change the amount of timesteps the models should be trained for via a command line argument

## About Reinforcement Learning and the Models

Reinforcement Learning (RL) is a subfield of machine learning that focuses on training agents to make optimal decisions in dynamic environments. 
RL relies on a trial-and-error approach, where agents learn through interactions with the environment, receiving feedback in the form of rewards. 
RL agents must balance between exploring new actions and exploiting their current knowledge to maximize long-term rewards. 
As they get better at getting rewards from their environments, we expect the reward for each episode to generally increase, if the model learns.


The three models we provide tackle the learning in different ways. 
Here is some very brief information on them:

#### Proximal Policy Optimization (PPO):
Proximal Policy Optimization (PPO) belongs to the family of policy gradient methods. 
PPO aims to optimize the policy of an agent by iteratively updating it while maintaining the stability of the learning process. 
PPO achieves this by limiting the magnitude of policy updates, ensuring more consistent and reliable improvements. 
Because it always learns from a batch of episodes, you will see the reward over timesteps have little "jumps" in the visualization.
Rewards will also not be mapped from zero onwards. 
This is not an error, but an artifact corresponding to the batch size.

#### Advantage Actor-Critic (A2C):
Advantage Actor-Critic (A2C) is a popular variant of the Actor-Critic algorithm.
A2C uses a neural network-based actor to choose actions and a critic to estimate the state-action value function. 
By computing advantages to guide policy updates, A2C reduces the variance in learning and promotes faster convergence. 
This approach facilitates parallelization, enabling A2C to efficiently leverage modern hardware resources, making it suitable for large-scale RL applications and real-time decision-making in challenging environments.
We do not utilize the full potential of this in this small demonstration of course, as we only run one instance.


#### Deep Q Networks (DQN):
Deep Q Networks (DQN) revolutionized Reinforcement Learning by introducing deep neural networks to approximate the Q-value function. 
DQN is a form of value iteration method that leverages a neural network as a function approximator to estimate the Q-values of state-action pairs. 
By using experience replay and a target network, DQN stabilizes learning and significantly improves convergence. 
DQN with the default hyperparameters is instantiated with a very small neural network, whereas the optimized version uses a network of size 256x256.
It would take longer than the 10k timesteps we provide to train the network to optimally get rewards. 
The RL baselines3 Zoo benchmark shows an average reward of 500 (the maximum for this environment) after 50k timesteps, showing the effectiveness of this approach.

## About CartPole
The CartPole environment is a classic benchmark problem in Reinforcement Learning (RL), designed to test an agent's ability to balance a pole on a moving cart. 
The agent can apply two discrete actions: pushing the cart left or right. 
The goal is to keep the pole balanced for as long as possible, while avoiding the pole from falling beyond a threshold angle or the cart moving out of bounds.
Also, the environment is limited to a maximum of 500 timesteps, and since each timestep in which the pole has not fallen yields a reward of 1, a maximum reward of 500 can be achieved.

## Authors

- [Christian Meißner](https://github.com/christian-meissner) - chrmeissner@uos.de
- [Kamran Vatankhah](https://github.com/kamranvat) - kvatankhahba@uos.de




