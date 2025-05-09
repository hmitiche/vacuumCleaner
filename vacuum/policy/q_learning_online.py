from .base import CleanPolicy
from ..maps import Map
from ..world import VacuumCleanerWorldEnv
from tools import Tools
import os
import pickle
import numpy as np


class QLearnOnlinePolicy(CleanPolicy):

    def __init__(self, world_id, env):
        super().__init__("q-learning-online", world_id, env)
        self.q_table = None
        self._rng = np.random.default_rng()
        self.map_dimension = self.env.unwrapped.map_size
        self.learning_rate = 0.2
        self.discount_factor = 0.95
        self.epsilon = 0.05  # small epsilon for online learning (exploitation mostly)
        self.visit_counts = {}  # pour suivre le nombre de visites par position

        # initialiser Q-table
        num_states = self.map_dimension * self.map_dimension * 2
        self.q_table = np.zeros((num_states, env.action_space.n))

        # Charger Q-table si elle existe
        self.load_qtable()

    def encode_state(self, state):
        x, y, z = state['agent'][0], state['agent'][1], state['dirt']
        return (x * self.map_dimension + y) * 2 + int(z)

    def decode_state(self, index):
        z = bool(index % 2)
        y = (index // 2) % self.map_dimension
        x = (index // 2) // self.map_dimension
        return {"agent": np.array([x, y]), "dirt": z}

    def load_qtable(self):
        filename = f"data/qlearning_online_table_map_{self.world_id}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
                print(f"[info] Q-table (online) chargée pour la carte '{self.world_id}'")

    def save_qtable(self):
        filename = f"data/qlearning_online_table_map_{self.world_id}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
            print(f"[info] Q-table (online) sauvegardée pour la carte '{self.world_id}'")

    def select_action(self, state):
        state_index = self.encode_state(state)
        if self._rng.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state_index])

    def update_qtable(self, state, action, reward, next_state):
        state_index = self.encode_state(state)
        next_index = self.encode_state(next_state)
        best_next = np.max(self.q_table[next_index])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.learning_rate * td_error

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, observation):
        """
        Cette méthode est appelée automatiquement à chaque pas de simulation.
        Elle sélectionne une action et met à jour la Q-table.
        """
        action = self.select_action(observation)
        old_obs = observation.copy()

        # Exécute l'action
        new_obs, reward, terminated, truncated, info = self.env.step(action)

        # Calcule la position de l'agent après déplacement
        pos_tuple = tuple(new_obs["agent"])
        if pos_tuple not in self.visit_counts:
            self.visit_counts[pos_tuple] = 1
            reward += 2.0 # nouvelle case visitée
        else:
            self.visit_counts[pos_tuple] += 1
            reward -= 0.05 * np.log(self.visit_counts[pos_tuple])  # revisitée

        # Mise à jour de la Q-table
        self.update_qtable(old_obs, action, reward, new_obs)

        return action

