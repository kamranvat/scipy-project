import pandas as pd
import matplotlib.pyplot as plt

def compare(*logs):
    """
    This function takes a two csv log files as input and displays some performance properties of the previously ran models.

    Args:
        log1: the name of the first csv file containing a models performance data
        log2: the name of the second csv file containing a models performance data
    """

    """# reading the chosen csv files each as a panda dataframe 
    data1 = pd.read_csv(f"logs/{log1}.csv")
    data2 = pd.read_csv(f"logs/{log2}.csv")"""

    d = {}
    for k in range(len(logs)):
        d["data{0}".format(k)] = pd.read_csv(f"logs/{logs[k]}.csv")

    # implement comparison between mean rewards either visually or numerically

    # create a figure showing the mean episode reward over time for both models
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    for key in d:
        ax.plot(d[key]["time/total_timesteps"], d[key]['rollout/ep_rew_mean'], label=key)
        ax.set(ylabel="Mean Episode Rewards", xlabel="Number of Timesteps", title="Mean Episode Rewards depending on Timesteps")
        ax.legend()

    plt.show()

