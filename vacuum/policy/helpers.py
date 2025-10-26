"""
'vacuum.policy.helpers.py'
-----------
Vacuum cleaner world, 2024.
Helper functions as for listing available policies and 
creating a given cleaning policy
Hakim Mitiche
March 2024
"""

from vacuum.policy.base import RandomPolicy				# relative module naming
from vacuum.policy.qlearning_1 import QLearnPolicy
from vacuum.policy.greedy import GreedyPolicy
from vacuum.policy.greedyrandom import GreedyRandomPolicy
from constants import TRAINING_PLOT_PATH
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def make_policy(policy_id, world_id, env, eco_mode):
	"""
	Instantiates a cleaning policy
	Parameters:
		policy_id (Int): policy identifier, see: 'main.py'
		world_id (String): map identifier
		eco_mode (Boolean): flag for economic mode. 
							when set, the agent stops checking rooms 
							if told that dirt won't comeback.
	Return: a cleaning policy
	"""
	pdict = get_policies()
	match (pdict[policy_id]):
		case 0:	p = RandomPolicy(env)
		case 1: p = GreedyRandomPolicy(world_id, env, eco=eco_mode)
		case 2:	p = GreedyPolicy(world_id, env, eco=eco_mode)
		case 3:	p = QLearnPolicy(world_id, env)
		case _:
			raise ValueError(f"Incorrect policy identifier {policy_id}!")
			#self.logger.critical("Incorrect policy id!")
			return
	return p


def get_policies():
	"""
	Returns a dictionary with cleaning policies names as keys and 
	policies identifying number as values. These are the policies 
	you defined in vacuum.policy based CleanPolicy in 'base.py'
	"""
	return{
		"random": 0,			# pure random, reflex-based agent
		"greedy-random": 1,		# greedy with some randomness, reflex-based agent
		"greedy": 2,			# greedy, usually a relex-based agent with model	
		"q-learning": 3,		# Q-learning, a learning-based agent
	}

def plot_epsilon(episodes, epsilon_log, decay_rate, policy_name="QL", map_name="Grid", save=False):
    """
    Plot epsilon decay over training episodes.

    Parameters:
        episodes (int): Total number of episodes.
        epsilon_log (array(float)): epsilon values.
        decay_rate(float): epsilon decay rate value
        policy_name (str): Name of the policy (for title/label).
        map_name (str): Name of the map/environment.
        save (bool): Whether to save the plot as a file.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(range(episodes), epsilon_log, label=f"{policy_name} ε-decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title(f"Epsilon Decay - {policy_name} on {map_name} (decay={decay_rate})")
    plt.grid(True)
    plt.legend()

    if save:
        filename = f"{TRAINING_PLOT_PATH}epsilon_decay_{policy_name}_{map_name}.png"
        plt.savefig(filename)
        print(f"[info] Plot saved as '{filename}'")

    plt.show()


# Improved smoothing function with edge handling
def smooth(data, window=50):
    if window < 1:
        raise ValueError("Window size must be at least 1")
    data = np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='same')

def plot_smooth_training_results(policy_id, map_id, rewards, travels, cleanings, window_size=50):
    # Smooth data
    smoothed_rewards = smooth(rewards, window=window_size)
    smoothed_travels = smooth(travels, window=window_size)
    smoothed_cleanings = smooth(cleanings, window=window_size)

    # Create DataFrames
    episodes = np.arange(len(smoothed_rewards))
    df = pd.DataFrame({
        'Episode': episodes,
        'Smoothed Reward': smoothed_rewards,
        'Smoothed Travel': smoothed_travels,
        'Smoothed Cleaning': smoothed_cleanings
    })

    # Plotting
    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    sns.lineplot(data=df, x='Episode', y='Smoothed Reward', color='dodgerblue')
    plt.title(f'Smoothed Rewards (Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 3, 2)
    sns.lineplot(data=df, x='Episode', y='Smoothed Travel', color='mediumseagreen')
    plt.title(f'Smoothed Travel Steps (Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 3, 3)
    sns.lineplot(data=df, x='Episode', y='Smoothed Cleaning', color='tomato')
    plt.title(f'Smoothed Cleaning Actions (Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Cleans')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Ask user if they want to save the figure
    save = input("[prompt] Do you want to save this plot? (y/n): ").strip().lower()
    if save == 'y':
        #filename = input("Enter filename (without extension): ").strip()
        filename = TRAINING_PLOT_PATH + "oneplot_results_"+ policy_id 
        + "_map_" + map_id
        if filename:
            path = f"{filename}.png"
            plt.savefig(path)
            print(f"[info] Plot saved as '{path}'")
        else:
            print("[error] Filename was empty. Plot not saved!")

    plt.show()
