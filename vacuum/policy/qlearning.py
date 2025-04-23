#from .base import CleanPolicy

#class QLearnPolicy(CleanPolicy):

	#def __init__(self, wolrd_id, env):
	#	raise NotImplementedError

	#def select_action(self, state):
		#raise NotImplementedError
from .base import CleanPolicy
from ..maps import Map
from ..world import VacuumCleanerWorldEnv
import os
import pickle
import numpy as np

class QLearnPolicy(CleanPolicy):

    def __init__(self, world_id, env):
        super().__init__("q-learning", world_id, env)
        self.q_table = None
        self.trained = False
        self.epsilon = 1.0  # Valeur initiale d'epsilon
        self.map_size =3  # Assure-toi que grid_size est défini dans l'env

    def reset(self, seed=None):
        pass

    def save_qtable(self):
        filename = f"data/qlearning_table_map_{self.world_id}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_qtable(self):
        filename = f"data/qlearning_table_map_{self.world_id}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            self.trained = True
            return True
        return False

    def encode_observation(self, observation):
        x, y = observation["agent"]
        dirt = observation["dirt"]
        return x * self.map_size * 2 + y * 2 + dirt

    def select_action(self, state_index):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # exploration
        else:
            return np.argmax(self.q_table[state_index])  # exploitation

    def train_q_learning(self, env, episodes, learning_rate_a=0.1, discount_factor_g=0.95,
                         epsilon=1.0, epsilon_min=0.01, epsilon_decay_rate=0.001):
        num_states = self.map_size * self.map_size * 2  # 2 = dirt (0 or 1)
        self.q_table = np.zeros((num_states, env.action_space.n))
        self._episode_reward = 0
        self.epsilon = epsilon  # Mettre à jour epsilon pour select_action()

        for episode in range(episodes):
            obs = env.reset()[0]
            state_index = self.encode_observation(obs)
            terminated = truncated = False

            while not terminated and not truncated:
                action = self.select_action(state_index)
                new_obs, reward, terminated, truncated, info = env.step(action)
                new_state_index = self.encode_observation(new_obs)

                # Q-learning update
                best_next_action = np.max(self.q_table[new_state_index])
                self.q_table[state_index][action] += learning_rate_a * (
                    reward + discount_factor_g * best_next_action - self.q_table[state_index][action]
                )

                self._episode_reward += reward
                state_index = new_state_index

            self.epsilon = max(epsilon_min, self.epsilon - epsilon_decay_rate)

        self.save_qtable()
        print(f"[info] Q-table sauvegardée dans data/qlearning_table_map_{self.world_id}.pkl")
