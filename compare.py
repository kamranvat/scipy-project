import pandas as pd
import matplotlib.pyplot as plt

def compare(log1,log2):
    """
    This function takes a two csv log files as input and displays some performance properties of the previously ran models.

    Args:
        log1: the name of the first csv file containing a models performance data
        log2: the name of the second csv file containing a models performance data
    """

    # reading the chosen csv files each as a panda dataframe 
    data1 = pd.read_csv(f"logs/{log1}.csv")
    data2 = pd.read_csv(f"logs/{log2}.csv")

    print(""" 
        The environment describes the world the agent is located in and changes its state based 
        on the behavior of the agent. The learning process is based on the experience the agent 
        gains through exploring the environment by executing different actions and receiving 
        feedback (rewards) from the environment depending on how good or bad the chosen actions 
        were. The reward is therefore an important measure when evaluating a models performance.\n""")
    
    # create a figure showing the mean episode reward over time for both models
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(data1["time/total_timesteps"], data1['rollout/ep_rew_mean'], label=f"{log1}")
    ax.plot(data2["time/total_timesteps"], data2['rollout/ep_rew_mean'], label=f"{log2}")
    ax.set(ylabel="Mean Episode Rewards", xlabel="Number of Timesteps", title="Mean Episode Rewards depending on Timesteps")
    ax.legend()

    plt.show()



