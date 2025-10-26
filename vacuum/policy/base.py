"""
base.py
-------
The base class to define an agent policy that vacuum cleans 
a grid like map.
Subclasses (policies): random (here), best-i-know (greedy) 
and QL (QlearnPolicy defined in 'qlearning.py').
The first and second policies should be used as baselines 
to evaluate the last policy. All serve to demonstrate 
some sort of intelligent agents in the AI class I teach 
to Computer Science undergraduates.

Author: Hakim Mitiche
Date: March 2024
"""

import numpy as np
#import random
import logging
import os, os.path
from abc import ABC, abstractmethod


LOG_PATH = "log/"


class CleanPolicy(ABC):
	"""
	The Base class to define a vacuum cleaning strategy.
	It can't be instantiatied.
	"""
	def __init__(self, policy_id, world_id, env, eco_mode=False):
		"""
		The eco_mode is a flag that tells the agent that dirt 
		won't comeback. The agent can then shift to economic mode 
		after visiting (cleaning) all rooms.
		"""
		self.policy_id = policy_id			# policy id name
		self.world_id = world_id
		self.env = env
		self.eco_mode = eco_mode
		#@CHECK: remove me later
		assert world_id == self.env.unwrapped.map_name
		#self._action_space = env.action_space
		self._action_dict = self._get_action_dict()
		self._location_sensor = env.unwrapped.location_sensor
		logfile = f"{LOG_PATH}{world_id}-{policy_id}.log"
		if os.path.isfile(logfile):
			# remove file to avoid large jammed files
			os.remove(logfile)
		logging.basicConfig(filename=logfile, level=logging.DEBUG)
		self.logger = logging.getLogger(__name__)
		self._seeded = False

	@abstractmethod
	def select_action(self, state):
		""" 
		Selects a single action to do, based on the current observation.
		To define for each actual policy.
		:param: state: the env current state, as seen by the agent
		:return: an action from env.action_space.n
		"""	
		raise NotImplementedError


	def _get_action_dict(self):
		"""
		Returns the set of actions the vacuum cleaner can do, 
		as a dictionary {action_name: action_number}.
		@IMPROVE_ME: rather just inverse the dictionary action of 
		VacuumCleanWorld. Better choice for maintenance.
		"""	
		return{
			"none":0, "suck":1, 
			"down":2, "right":3, "up":4, "left":5
		}

	@abstractmethod
	def reset(self):
		"""
		Resets the policy parameters used during an episode.
		Must be implemented and called in the beginning of a 
		new episode
		"""	
		raise NotImplementedError


"""
A cleaning policy which is merely random.
"""
MY_NAME = "random"
@register_policy(MY_NAME)
class RandomPolicy(CleanPolicy):
	"""
	Pickup a random action at each step. 
	The agent is purely random reflex-based. 
	This is the very basic baseline in any problem.
	Parameters:
		env (gym.env): the agent environment
		seed (Float): a seed (number) to initialize the random number 
					  generator used by the policy. can be that of 
					  the env?!
	"""
	def __init__(self, world_id, env):
		super().__init__(MY_NAME, world_id, env)
		self.nbr_actions = env.action_space.n
		self._rng = None # random number generator
		self._seeded = False

	def select_action(self, state):
		"""
		selects an action randomly
		"""
		#assert self._rng is not None, f"[error] there isn't a RNG!\
		#'{__class__.__name__}' needs a reset before use!"
		return self._rng.choice(self.nbr_actions)


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