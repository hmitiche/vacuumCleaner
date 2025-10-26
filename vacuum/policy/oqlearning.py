
from .base import CleanPolicy
from ..maps import Map
from ..world import VacuumCleanerWorldEnv
from tools import Tools
import os
import pickle
import numpy as np
from tqdm import tqdm
import numpy as np

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.95

class OQLearnPolicy(CleanPolicy):
    """
    Online qlearning
    """

    def __init__(self, world_id, env):
        super().__init__("oq-learning", world_id, env)
        self._rng = None # random number generator
        self._seeded = False
        self.map_dimension = n = self.env.unwrapped.map_size
        self.q_table = None
        self.visits = None

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
            #print("[debug] np.random seeded with ", seed)
            self._seeded = True
        self.visits_level = np.zeros((n,n))   # level of number of visits per tile
        # free parameters initialization
        self.temperature = 1.0
        self.epsilon = 0.95
        self.alpha = 0.2

    def encode_state(self, state):
        """
        Encode a state to optimize memory and lookup in Qtable
        """ 
        x,y,z = state['agent'][0], state['agent'][1], state['dirt']
        state_index = (x * self.map_dimension + y) * 2 + int(z)
        assert 0 <= state_index < self.q_table.shape[0]
        return state_index

    def decode_state(self, index):
        """
        Decode a state index to the corresponding state
        :param index: the agent state (observation) index 
        obtained from self.encode_state()
        :return: a Dict() with keys: 'agent' (location), 'dirt' (boolean)
        """
        z = bool(index % 2)
        y = index % self.map_dimension
        x = index // self.map_dimension
        return {"agent": np.array([x, y]), "dirt": z}

    def update_freeparameters(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
            
    def select_action(self, state):
        state_index = encode_state(state)
        action = softmax_action_selection(state_index, self.temperature)

    def softmax_action_selection(state_index, temperature=1.0):
        # stochastic action selection based on Q values and 
        # temperature
        s = self.encode_state(state)
        q_values = q_table[state]
        exp_q = np.exp(q_values / temperature)
        probabilities = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probabilities)