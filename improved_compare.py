import pandas as pd
import matplotlib.pyplot as plt

def compare(model_list):
    """
    This function dictionary of models as input and displays some performance properties of the previously ran models.

    Args:
        model_list (list of dicts): 
            each dict contains information about one model,
            (name, description, active)

    """
    logs = read_logs(model_list)

    print(""" 
        The environment describes the world the agent is located in and changes its state based 
        on the behavior of the agent. The learning process is based on the experience the agent 
        gains through exploring the environment by executing different actions and receiving 
        feedback (rewards) from the environment depending on how good or bad the chosen actions 
        were. The reward is therefore an important measure when evaluating a models performance.\n""")

    # create a figure showing the mean episode reward over time for both models
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    # iterate over the list of logs
    for log in logs:
        ax.plot(log["time/total_timesteps"], log['rollout/ep_rew_mean'], label=log)
        ax.set(ylabel="Mean Episode Rewards", xlabel="Number of Timesteps", title="Mean Episode Rewards depending on Timesteps")
        ax.legend()

    plt.show()


def read_logs(model_list):
    """
    Opens the .csv files for each active model in model_list and returns them as list

    Args:
        model_list (list of dicts): 
            each dict contains information about one model,
            (name, description, active)


    Returns:
        logs_list: list of the logs of active models, from the logs folder
    """

    logs_list = []

    for model in model_list:
        if model.get("active"):
            model_name = model.get("name")
            logs_list.append(pd.read_csv(f"logs/{model_name}.csv"))

    return logs_list

