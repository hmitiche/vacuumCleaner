"""
Control policies for Vacuum Cleaner World problem
we define 3 decision policies: random, best-i-knwo and 
QL (defined as QlearnerPolicy in vacuum_cleaner_policy)
Hakim Mitiche
March 2024
"""

import numpy as np
#import random
import logging
import os, os.path

LOG_PATH = "log/"

"""
Base class for a vacuum cleaning policy
"""
class CleanPolicy():

	def __init__(self, policy_id, world_id, env):
		self.policy_id = policy_id
		self.world_id = world_id
		self.env = env
		#self._action_space = env.action_space
		self._action_dict = self._get_action_dict()
		self._location_sensor = env.unwrapped.location_sensor
		logfile = f"{LOG_PATH}{world_id}-{policy_id}.log"
		if os.path.isfile(logfile):
			os.remove(logfile)  
		logging.basicConfig(filename=logfile, level=logging.DEBUG)
		self.logger = logging.getLogger(__name__)
		self._seeded = False

	""" Selects a single action to do, based on current env observation.
		:param: state: current env state as seen by the agent
		:return: an action in env.action_space.n
	"""		
	def select_action(self, state):
		raise NotImplementedError

	"""
	Vacuum cleaner actions dictionary
	{action_name: action_number} 
	"""	
	def _get_action_dict(self):
		return{
			"none":0, "suck":1, 
			"down":2, "right":3, "up":4, "left":5
		}

	"""
	Resets the policy parameters used during an episode.
	Must be called, if any, at the beginning of a new episode
	"""	
	#@abtractmethod
	def reset(self):
		raise NotImplementedError


"""
Random cleaning policy.
An pure random reflex-based agent program.
The very basic decision policy for a vacuum cleaner agent.
"""
class RandomPolicy(CleanPolicy):
	"""
	Pickup a random action at each step.
	very weak baseline for QL and 'best-i-know' policies
	env (gym.env) the agent environment
	"""
	def __init__(self, env, seed=None):
		self.policy_id = "random"
		self.nbr_actions = env.action_space.n
		self._rng = None
		self._seeded = False

	def select_action(self, state):
		assert self._rng is not None, "please reset the policy before use!"
		return self._rng.choice(self.nbr_actions)

	"""
	Seed the RNG, only once, unless a seed is provided 
	during reset.
	"""	
	def reset(self, seed=None):
		if not self._seeded or seed is not None:
			self._rng = np.random.default_rng(seed)
			#print("[debug] np.random seeded with ", seed)
			self._seeded = True