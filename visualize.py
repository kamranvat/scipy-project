import pandas as pd
import matplotlib.pyplot as plt

def visualize(logs):
    """
    This function takes a csv log file as input and displays some performance properties of the previously ran models.

    Args:
        logs: the name of the csv file containing a models performance data
    """

    # reading the chosen csv file as a panda dataframe 
    data = pd.read_csv(f"logs/{logs}.csv")

    # display the dataframe
    print(f"This is an overview over the logged parameters from the {logs} run.")
    print(data.head())

    print("This is a visualization of the mean episode reward depending on elapsed time: /n")
    # create a figure showing the mean episode reward over time
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(data["time/total_timesteps"], data['rollout/ep_rew_mean'])
    ax.set(ylabel="mean episode reward", xlabel="time", title="mean episode reward depending on elapsed time")
    plt.show()




visualize("PPO")