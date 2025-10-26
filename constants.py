
# constants.py

__all__ = ["LOG_PATH", "DATA_PATH", "PLOT_PATH", "TRAINING_DATA_PATH", "TRAINING_PLOT_PATH",\
"EPISODES", "TRAIN_EPISODES", "STEPS", "SEED"]

# file pathnames
LOG_PATH = 'log/'
DATA_PATH = 'data/'
PLOT_PATH = 'plots/'
TRAINING_DATA_PATH = DATA_PATH + "training/"
TRAINING_PLOT_PATH = PLOT_PATH + "training/"
SIMULATION_DATA_PATH = DATA_PATH + "simulation_online/"
SIMULATION_PLOT_PATH =PLOT_PATH + "simulation_online/"

# gym environment constants
EPISODES = 1        		  # number of episodes (default value)
TRAIN_EPISODES = 1000		  # number of training episodes for QLearning policy
STEPS = 200                   # number of steps in epsiode (default value)
SEED = 0                      # RNG seed (default value), for env dynamics replication