# CartPole-v1 Performance Measurement

We are measuring and showing the performance of different policies in the CartPole-v1 Reinforcement Learning environment over a short amount of timesteps.

This project serves as the final project for the Scientific Python course at the University of Osnabrück, summer semester of 2023.


## Table of Contents

- [Goal](#goal)
- [Motivation](#motivation)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Authors](#authors)

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


### Explanation

### Command line arguments

### Saving and loading

## Features

- List the main features of your project.
- Feature 1
- Feature 2
- ...


## Authors

- [Christian Meißner](https://github.com/christian-meissner) - chrmeissner@uos.de
- [Kamran Vatankhah](https://github.com/kamranvat) - kvatankhahba@uos.de




