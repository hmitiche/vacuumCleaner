import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Improved smoothing function with edge handling
def smooth(data, window=50):
    if window < 1:
        raise ValueError("Window size must be at least 1")
    data = np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='same')

# Plotting function for training results
def _plot_training_results(rewards, travels, cleanings, window_size=50):
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
    save = input("Do you want to save this plot? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Enter filename (without extension): ").strip()
        if filename:
            path = f"{filename}.png"
            plt.savefig(path)
            print(f"Plot saved as '{path}'")
        else:
            print("Filename was empty. Plot not saved.")

    plt.show()

# Example data for testing
np.random.seed(42)
rewards = np.cumsum(np.random.randn(500) * 0.5 + 0.5)
travels = np.random.randint(10, 30, size=500)
cleanings = np.random.randint(5, 15, size=500)

# Call the plotting function
_plot_training_results(rewards, travels, cleanings)