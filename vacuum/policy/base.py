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


LOG_PATH = "log/"


class CleanPolicy():
	"""
	The Base class to define a vacuum cleaning strategy.
	"""
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
		as dictionary {action_name: action_number} 
		"""	
		return{
			"none":0, "suck":1, 
			"down":2, "right":3, "up":4, "left":5
		}

	#@abtractmethod
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
	def __init__(self, env, seed=None):
		self.policy_id = "random"
		self.nbr_actions = env.action_space.n
		self._rng = None
		self._seeded = False

	def select_action(self, state):
		assert self._rng is not None, f"please reset the policy before \
		can use {__class__.__name__}!"
		return self._rng.choice(self.nbr_actions)


	def reset(self, seed=None):
		"""
		Seeds the RNG only once, unless a seed is provided 
		during reset.
		Parameters:
			seed (int): default value none
		"""	
		if not self._seeded or seed is not None:
			self._rng = np.random.default_rng(seed)
			#print("[debug] np.random seeded with ", seed)
			self._seeded = True