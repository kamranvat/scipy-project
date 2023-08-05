import pandas as pd
import matplotlib.pyplot as plt

def compare(model_list):
    """
    This function takes a two csv log files as input and displays some performance properties of the previously ran models.

    Args:
        model_list (list of dicts): 
            each dict contains information about one model,
            (name, description, active)

    """

    """# reading the chosen csv files each as a panda dataframe 
    data1 = pd.read_csv(f"logs/{log1}.csv")
    data2 = pd.read_csv(f"logs/{log2}.csv")"""

    logs = read_logs(model_list)

    for log in logs:
        # create a figure showing the mean episode reward over time for both models
        fig, ax = plt.subplots(nrows = 1, ncols = 1)

        #for key in log:
        #    print(key)
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



