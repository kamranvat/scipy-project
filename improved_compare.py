import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean 

def compare(model_list):
    """
    This function dictionary of models as input and displays some performance properties of the previously ran models.

    Args:
        model_list (list of dicts): 
            each dict contains information about one model,
            (str name, str description, bool active, dict model_args)

    """
    # extract the logged performance measurements and model titles seperately from the dictionary
    logs, titel = read_logs(model_list)

    # creation of a variable for iteration over the model titels  
    model_num = 0
    
    # creation of a list for the models average rewards
    avg_reward = []
    for log in logs:
        print(f"\n\n\nThis is an overview over the logged parameters from the {titel[model_num]} run:\n")
        print(log.head())
        avg_reward.append(mean(log["rollout/ep_rew_mean"])) 
        print(f"\n{titel[model_num]}s average reward over all timesteps is {avg_reward[model_num]}.")
        model_num += 1

    print(""" 
        The environment describes the world the agent is located in and changes its state based 
        on the behavior of the agent. The learning process is based on the experience the agent 
        gains through exploring the environment by executing different actions and receiving 
        feedback (rewards) from the environment depending on how good or bad the chosen actions 
        were. The reward is therefore an important measure when evaluating a models performance.\n""")

    # create a figure showing the mean episode reward over time for both models
    fig, ax = plt.subplots(nrows = 1, ncols = 1)

    # reset of the iteration variable
    model_num = 0 
    # iterate over the list of logs and display them in one plot
    for log in logs:
        ax.plot(log["time/total_timesteps"], log['rollout/ep_rew_mean'], label=titel[model_num])
        ax.set(ylabel="Mean Episode Rewards", xlabel="Number of Timesteps", title="Mean Episode Rewards depending on Timesteps")
        model_num += 1
        ax.legend()    
    plt.xlim(left=0)
    plt.show()


def read_logs(model_list):
    """
    Opens the .csv files for each active model in model_list and returns them as list

    Args:
        model_list (list of dicts): 
            each dict contains information about one model,
            (str name, str description, bool active, dict model_args)


    Returns:
        logs_list: list of the logs of active models, from the logs folder
        titel_list: list of model titles
    """

    logs_list = []
    titel_list = []

    # iterate over the active models in the dictionary and get the logs and names
    for model in model_list:
        if model.get("active"):
            model_name = model.get("name")
            logs_list.append(pd.read_csv(f"logs/{model_name}.csv"))
            titel_list.append(model_name)

    return logs_list, titel_list

