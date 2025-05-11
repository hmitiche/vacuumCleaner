from .base import CleanPolicy
from ..maps import Map
from ..world import VacuumCleanerWorldEnv
from tools import Tools
import os
import pickle
import numpy as np
from tqdm import tqdm


class QLearnPolicy(CleanPolicy):

    def __init__(self, world_id, env):
        super().__init__("q-learning", world_id, env)
        self.trained = False
        self.episodes=2000
        self.q_table = None
        self._rng = None  # random number generator
        self._seeded = False
        self.map_dimension = self.env.unwrapped.map_size

    def reset(self, seed=None):
        """
        Resets the policy by seeding the RNG.
        When no seed is given, RNG is seeded once.
        The RNG is reseeded if a seed value is provided
        Parameters:
            seed (int): default value is None
        """
        if not self._seeded or seed is not None:
            self._rng = np.random.default_rng(seed)
            # print("[debug] np.random seeded with ", seed)
            self._seeded = True

    def load_qtable(self):
        """
        @HM: load qtable and return True or return False
        if the agent hasn't been trained yet"
        """
        filename = f"data/qlearning_table_map_{self.world_id}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
                print(f"[info] Q-table chargée pour la carte '{self.world_id}'")
                f.close()  # @HM: close the file you opened
            self.trained = True
            return True
        return False

    def save_qtable(self):
        filename = f"data/qlearning_table_map_{self.world_id}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
            f.close()  # @HM: close the file you opened
            print(f"[info] Q-table saved in 'data/qlearning_table_map_{self.world_id}.pkl'")

    def encode_state(self, state):
        """
        encode a state to optimize memory and lookup in Qtable
        """
        x, y, z = state['agent'][0], state['agent'][1], state['dirt']
        state_index = (x * self.map_dimension + y) * 2 + int(z)
        assert 0 <= state_index < self.q_table.shape[0]
        return state_index

    def decode_state(self, index):
        """
        decode a state index to the corresponding state
        :param index: the agent state (observation) index
        obtained from self.encode_state()
        :return: a Dict() with keys: 'agent' (location), 'dirt' (boolean)
        """
        z = bool(index % 2)
        y = index % self.map_dimension
        x = index // self.map_dimension
        return {"agent": np.array([x, y]), "dirt": z}

    def select_action_training(self, state_index):
        """
        How to select an action during training: ε-greedy policy
        :param state_index: current agent state index in QL table
        """
        if self._rng.uniform(0, 1) < self.epsilon:
            # Exploration: choisir une action aléatoire
            return self.env.action_space.sample()
        else:
            # Exploitation: choisir l'action estimée ma meilleiur (valeur Q maximale)
            return np.argmax(self.q_table[state_index])

    def select_action(self, state):
        # selects the action estimated best (according qtable)
        state_index = self.encode_state(state)
        return np.argmax(self.q_table[state_index])  # exploitation

    # train Q-Learning agent with Epsilon-Greedy and Linear Epsilon Decay
    def train_q_learning(self, env, learning_rate_a=0.2, discount_factor_g=0.95,
                         epsilon=1.0, epsilon_min=0.05, epsilon_decay_rate=0.001):

        # compute the number of QL table entries

        num_states = self.map_dimension * self.map_dimension * 2  # 2 = a room is dirty or clean
        self.q_table = np.zeros((num_states, env.action_space.n))  # init empty QL table
        self._episode_reward = 0
        self.epsilon = epsilon
        rewards = np.zeros(self.episodes)
        cleanings = np.zeros(self.episodes)
        travels = np.zeros(self.episodes)

        for episode in tqdm(range(self.episodes), desc="Training", unit="ep"):
            obs = env.reset()[0]
            state_index = self.encode_state(obs)
            terminated = truncated = False

            #visited_positions = set()  # Pour garder trace des positions visitées dans cet épisode
            episode_reward = 0  # Récompense totale pour cet épisode

            visit_counts = {}

            while not terminated and not truncated:
                pos = tuple(obs['agent'])
                visit_counts[pos] = visit_counts.get(pos, 0) + 1

                action = self.select_action_training(state_index)
                new_obs, reward, terminated, truncated, info = env.step(action)
                new_position = tuple(new_obs['agent'])
                visit_counts[new_position] = visit_counts.get(new_position, 0) + 1

                # Récompense bonus ou pénalité
                if visit_counts[new_position] == 1:
                    reward += 2.0
                else:
                    reward -= 0.2 * np.log(visit_counts[new_position])

                episode_reward += reward

                # Mise à jour Q-table
                new_state_index = self.encode_state(new_obs)
                best_next_action = np.max(self.q_table[new_state_index])
                self.q_table[state_index][action] += learning_rate_a * (
                        reward + discount_factor_g * best_next_action - self.q_table[state_index][action]
                )

                # Mise à jour état
                state_index = new_state_index
                obs = new_obs

            # Enregistrement des statistiques de l'épisode
            rewards[episode] = episode_reward
            cleanings[episode] = env.get_wrapper_attr('_total_cleaned')
            travels[episode] = env.get_wrapper_attr('_total_travel')
            self.epsilon = max(epsilon_min, self.epsilon - epsilon_decay_rate)

        # show performance statistics
        print("[info] training avg. reward: ", round(sum(rewards) / self.episodes, 2))
        print("[info] training avg. cleaning: ", round(sum(cleanings) / self.episodes, 1))
        print("[info] training avg. travel distance: ", round(sum(travels) / self.episodes, 1))
        if input("[prompt] save training results? [y/n]") == 'y':
            results = {'reward': rewards, 'cleaned': cleanings, 'travel': travels}
            Tools.save_training_results(self.world_id, self.policy_id, results)

        self.save_qtable()

