
"""
QL with an aguemented state: position, dirt, last_visit.
The assumption is that is sufficiant for small environement 
for which I coded with success a reflex-based agent with model 
(GreedyPolicy in 'greedy.py')

NB: there are correctness assertion to remove after test!
author: Hakim Mitiche
date: May 2025
"""
from vacuum.policy.base import CleanPolicy
from vacuum.world import VacuumCleanerWorldEnv
from vacuum.maps import Map
#from tools import Tools as tl
from tools import Tools
import gymnasium as gym
from gym.wrappers import TimeLimit
import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# policy constants
TRAIN_EPISODES = 2000         # number of training episodes for QLearning policy
# should dependent on the map size and complexity?!
EPISODE_STEPS = 300           # number of steps per episode (proportional to difficulty) 
LEARNING_RATE_ALPHA = 0.2
DISCOUNT_FACTOR_G = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.1
# for exponential epsilon decay
EPSILON_DECAY_RATE = 0.95   # Medium decay — good for balance exploitation/exploration
# select .9 for fast decay, 0.97+ for slow decay (more exploration)
EPSILON_DECAY_AMOUNT = 0.01  # or another suitable small value (for linear decay)



class QLearnPolicy(CleanPolicy):

    def __init__(self, world_id, env):
        super().__init__("q-learning", world_id, env)
        self.trained = False
        self.q_table = None
        self._rng = None    # random number generator
        self._seeded = False
        self.map_dimension = self.env.unwrapped.map_size 

    def reset(self, seed=None):
        """
        Resets the policy by seeding the RNG. To call before 
        training and operation.
        When no seed is given, RNG is seeded once. 
        The RNG is reseeded if a seed value is provided
        Parameters:
            seed (int): default value is None
        """ 
        if not self._seeded or seed is not None:
            self._rng = np.random.default_rng(seed)
            #print("[debug] np.random seeded with ", seed)
            self._seeded = True
        n = self.map_dimension 
        # init room visits count and levels
        #self.visits_level = np.zeros((n,n))
        self.visits = np.zeros((n, n), dtype=np.int16)
        # self.visits[x,y] correspond to room with (x,y) coordinates
        if not self.trained:
            # to normalize the reward to used to build Q table
            self.rewards_list = []
            self.reward_mean = 0.0
            self.reward_std = 1.0

    def load_qtable(self):
        """
        Load Q table into {self.q_table} and return True or return False 
        if the agent hasn't been trained yet or the q table is corrupt 
        or missing
        """
        filename = f"data/qlearning_table_map_{self.world_id}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
                print(f"[info] Q-table loaded for map: '{self.world_id}'")
                #f.close()       # @HM: close the file you opened
                self.trained = True
        else:
            self.trained = False
        return self.trained

    def _save_qtable(self):
        """
        Store Q table, outputed by training, in a binary file 
        named after the the map identifier
        """
        filename = f"data/qlearning_table_map_{self.world_id}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
            # f.close()   # @HM: close the file you opened (done by with)
            print(f"[info] Q-table saved in 'data/qlearning_table_map_{self.world_id}.pkl'")

    def _encode_state(self, state):
        """
        Encode a state to optimize memory and lookup in Qtable, 
        by compute a unique index:
        (agent_y * map_width + agent_x) → flatten 2D grid
        * VISITS_CAP → add visit level (z) 
        * 2 + dirt → binary for dirty/clean
        """ 
        x,y,w = state['agent'][0], state['agent'][1], state['dirt']
        if 'visits_level' in state:
            z = state['visits_level']
        else:    
            z = self._get_visit_level(state)
        state_index = ((y * self.map_dimension + x) * VISITS_CAP+z) * 2 + int(w)
        return state_index

    def _decode_state(self, index):
        """
        Decode a state index into an agent state (for Q update)
        :param index: the agent state (observation) index 
        obtained from self.encode_state()
        :return: a Dict() with keys: 'agent' (location), 'dirt' (boolean)
        """
        w = bool(index % 2)
        index = index // 2
        z = index % VISITS_CAP 
        index = index // VISITS_CAP
        x = index % self.map_dimension
        y = index // self.map_dimension
        return {"agent": np.array([x, y], dtype=np.int16), "dirt": w, "visits_level":z}

    def _check_index(self, state_index):
        """
        Check if the state encoded for Q table is correct.
        For debug purpose.
        Parameter:
            state_index: index obtained from encoding the agent state.
        """
        assert 0 <= state_index < self.q_table.shape[0], f"State index {state_index} out of bound!"
        assert self._encode_state(self._decode_state(state_index)) == state_index,\
        f"state encoding mismatches decoding!"

    def _get_visit_level(self, state):
        """
        Returns a caped room visit level which is proportional to the 
        number of visits.
        """
        x, y = state['agent'][0], state['agent'][1]
        count = self._get_visit_count(x, y)
        return count if count < VISITS_CAP else VISITS_CAP-1

    def _get_visit_count(self, x, y):
        """
        Returns the number of visits to room (x,y), 
        where: x,y are respectively in X-axis, Y-axis
        """
        return self.visits[x, y]

    def _update_visit_count(self, obs, last_room):
        """
        Updates the number of visits to current room. 
        Updates the last visited room if necessary.
        Parameters:
            obs: current observation (agent, dirt)
            last_room: the room visited recently
        Return: the last visited room.
        """
        # current room coordinates
        x, y = obs['agent'][0], obs['agent'][1]
        if (x,y) != last_room:
            # a new visit
            self.visits[x,y] += 1
            return (x,y)            # update last_room
        return last_room

    #def update_freeparameters(self):
        """
        to call during training, 
        updates epsilon, the exploration/exploitation probability
        and possibly the learning rate (alpha)
        """
    def _epsilon_exponential_decay(self, epsilon, eps, nbr_eps):
        """
        Exponentionally decay epsilon independantly of 
        the number of episodes (nbr_eps), still this should be multiple 
        of 100. Epsilon decays following a curve that only depend on 
        EPSILON_DECAY_RATE value.
        Parameters:
            epsilon: current value
            eps: episode number
            nbr_eps: total number of episodes
        Return:
            the new value of epsilon
        """
        m = nbr_eps // 100
        if eps % m == 0:
            return max(EPSILON_MIN, epsilon * EPSILON_DECAY_RATE)
        else:
            return epsilon

    def _epsilon_linear_decay(self, epsilon):
        return max(EPSILON_MIN, epsilon - EPSILON_DECAY_AMOUNT)

    def _adjust_reward(self, reward, obs, last_room):
        """
        Adjust and normalized the reward to improve QL training.
        Modifies the reward obtained from the environment, so
        to encourage first-time visits more strongly, and gradually 
        penalize re-visits. To call before updating Q table. I guess 
        I need to make sure the agent did yet move out the room?!
        Parameters:
            reward environment reward
            obs current observation
            last_room the last visited room
        Return: a new reward value
        """
        x, y = obs['agent']
        if last_room == (x,y):
            return reward     # no updates

        # adjust reward for Q table    
        visit_penalty = np.log1p(self.visits[x, y])  # smoother than linear
        reward += 1.0 / (1.0 + self.visits[x, y])    # decaying bonus for new-ish visits
        reward -= 0.2 * visit_penalty
        
        # Update reward stats
        self.rewards_list.append(reward)
        if len(self.rewards_list) > 1:
            self.reward_mean = np.mean(self.rewards_list)
            self.reward_std = np.std(self.rewards_list) + 1e-8  # prevent divide by zero
        else:
            self.reward_mean = reward
            self.reward_std = 1.0

        # Z-score normalization
        reward = (reward - self.reward_mean) / self.reward_std
        return reward

    def _select_action_training(self, state_index, epsilon): 
        """
        Selects an action during training according to ε-greedy policy
        which gradually shifts the search tendancy from exploration to 
        exploitation.
        :param state_index: current agent state index in QL table
        """
        if self._rng.uniform(0, 1) < epsilon:
            # Exploration: choisir une action aléatoire
            return self.env.action_space.sample()
        else:
            # Exploitation: choisir l'action estimée ma meilleiur (valeur Q maximale)
            return np.argmax(self.q_table[state_index])
            
    def select_action(self, state):
        """
        The action selection function the QL agent uses during operation 
        to selects the action estimated best (according to Q table).
        """
        state_index = self._encode_state(state)
        return np.argmax(self.q_table[state_index])  # pure exploitation

    def train_q_learning(self, env, episodes=TRAIN_EPISODES, learning_rate_a=LEARNING_RATE_ALPHA, 
        discount_factor_g=DISCOUNT_FACTOR_G, epsilon=EPSILON):
        """
        Trains Q-Learning agent with Epsilon-Greedy and Eponential Epsilon Decay.
        reset() must be called before training. We let the agent even learn 
        by itself that it needs to clean a dirty room. The Q table is loaded 
        and ready to use afterwards.
        """
        # change the maximum steps per episode
        env = gym.wrappers.TimeLimit(env, max_episode_steps=EPISODE_STEPS)
        # compute the number of QL table entries
        num_states = self.map_dimension * self.map_dimension * VISITS_CAP * 2  
        # 2 = a room is dirty or clean
        self.q_table = np.zeros((num_states, env.action_space.n))   # init empty QL table
        # performance stats over episodes
        dirty_rooms = np.zeros(episodes)        # per episode initially
        rewards = np.zeros(episodes)
        cleanings = np.zeros(episodes)
        travels = np.zeros(episodes)
        # show training progress bar
        for episode in tqdm(range(episodes), desc="Training", unit="ep"): #range(episodes):
            obs, info = env.reset()
            dirty_rooms[episode] = info["dirty_spots"]
            last_room = self._update_visit_count(obs, None)
            state_index = self._encode_state(obs)
            self._check_index(state_index)
            done = False
            while not done:
                action = self._select_action_training(state_index, epsilon)
                obs, reward, terminated, truncated, info = env.step(action)
                new_state_index = self._encode_state(obs)
                #@DEBUG: comment me later 
                #self._check_index(state_index)
                # encourage new visits and penalize revist gradually
                """
                print(f"State: {obs}")
                print(f"State index: {self._encode_state(obs)}")
                print(f"Q-table shape: {self.q_table.shape}")
                """
                adjusted_reward = self._adjust_reward(reward, obs, last_room)
                reward = adjusted_reward
                last_room = self._update_visit_count(obs, last_room)
                # Q table update
                best_future_q = np.max(self.q_table[new_state_index])
                current_q = self.q_table[state_index, action]
                self.q_table[state_index, action] = (1 - learning_rate_a) * current_q + \
                learning_rate_a * (reward + discount_factor_g * best_future_q)
                state_index = new_state_index
                done = terminated or truncated
            epsilon = self._epsilon_exponential_decay(epsilon, episode, episodes)
            rewards[episode] = round(env.get_wrapper_attr('_episode_reward'), 2)
            cleanings[episode] = env.get_wrapper_attr('_total_cleaned')  
            travels[episode] = env.get_wrapper_attr('_total_travel')

        # show performance statistics
        print("[info] training avg. reward: ", round(sum(rewards)/episodes, 2))
        print("[info] training avg. cleaning: ", round(sum(cleanings)/episodes, 1))
        print("       avg. number of dirty rooms: ", round(np.mean(dirty_rooms)),1)
        print(" dirty rooms: ", dirty_rooms)
        print("[info] training avg. travel distance: ", round(sum(travels)/episodes, 1))
        if input("[prompt] Save training results? [y/n]") == 'y':
            results = {'reward':rewards, 'cleaned':cleanings, 'travel':travels}
            Tools.save_training_results(self.world_id, self.policy_id, results)
        
        self._save_qtable()
        # Optionally plot training metrics
        #self._plot_training_results(rewards, travels, cleanings)
        self.trained = True