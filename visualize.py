import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean 

def visualize(logs):
    """
    This function takes a csv log file as input and displays some performance properties of the previously ran models.

    Args:
        logs: the name of the csv file containing a models performance data
    """

    # reading the chosen csv file as a panda dataframe 
    data = pd.read_csv(f"logs/{logs}.csv")

    # display the dataframe
    print(f"This is an overview over the logged parameters from the {logs} run.\n")
    print(data.head())
    print(""" 
        The environment describes the world the agent is located in and changes its state based 
        on the behavior of the agent. The learning process is based on the experience the agent 
        gains through exploring the environment by executing different actions and receiving 
        feedback (rewards) from the environment depending on how good or bad the chosen actions 
        were. The reward is therefore an important measure when evaluating a models performance.\n""")
    
    # calculation of the mean reward 
    avg_reward = mean(data["rollout/ep_rew_mean"])
    print(f"Using {logs}, the average reward per episode is {avg_reward}.")
    
    #print("This is a visualization of the mean episode reward depending on total timesteps:")
    # create a figure showing the mean episode reward over time
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(data["time/total_timesteps"], data['rollout/ep_rew_mean'])
    ax.set(ylabel="Mean Episode Rewards", xlabel="Number of Timesteps", title="Mean Episode Rewards depending on Timesteps")
    plt.show()


