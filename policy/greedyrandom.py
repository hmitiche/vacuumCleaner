"""
Vacuum cleaner world, 2024.
'greedyrandom.py'
An reflex-based agent with random actions.
author: Hakim Mitiche
date: April, 6th 2024
"""

from policy.base import CleanPolicy
from maps import Map
import numpy as np
import random
import logging
#import os, os.path

""" 
Class for decision policy used by a random reflex-based agent.
Such agent is better then a pure random agent and should be less 
efficient then a 'best-i-knwo' reflex-based with model agent, 
but it's easier to code then the latter.
"""
class GreedyRandomPolicy(CleanPolicy):
	
	def __init__(self, world_id, env, eco=False):
		super().__init__("greedy-random", world_id, env)
		# map locations ordred: left->right, up->down
		# @see: 'maps.py'
		self._locations = Map.locations_list(world_id)
		assert self._locations is not None
		#self.reset(seed)
		#self.seeded = False

	"""
	Reset the policy. 
	:param: seed random number generator seed
	"""
	def reset(self, seed=None):
		if not self._seeded or seed is not None:
			random.seed(seed)
			print("[debug] random seeded with ", seed)
			self._seeded = True
	
	"""
	An implementation of CleanPolicy.select_action()
	Selects an action according to: current state, world map and policy
	"""
	def select_action(self, state):
		room = state['agent']			# agent current location
		dirty = state['dirt']			# is there dirt in current room?
		action = self.agent_program(room, dirty)
		return action

	""" 
	Call a random reflex-based agent program (function) 
	depending on the vacuum world geography.	
	"""
	def agent_program(self, room, dirty):
		match (self.world_id):
			case "vacuum-3rooms-v0":
				act = self.agent_3rooms_v0(room, dirty)
			case "vacuum-3rooms-v2":
				act = self.agent_3rooms_v2(room, dirty)
			case "vacuum-4rooms-v1":
				act = self.agent_4rooms_v1(room, dirty)
			case "vacuum-5rooms-v1":
				act = self.agent_5rooms_v1(room, dirty)
			case "vacuum-6rooms-v1":
				act = self.agent_6rooms_v1(room, dirty)
			case _:
				act = None
				#self.logger.critical("No agent program yet for ", self.world_id)
				print(f"[error] No agent program yet for {self.world_id} map")
				print(f"or the agent program is similar to \
					a pure reflex-based agent")
				exit()
		return act
		
	"""
	Random Reflex-based agent program for 'vacuum-3rooms-v0' world, 
	Check the map in 'maps.py'.
	"""	
	def agent_3rooms_v0(self, location, dirty):

		if dirty: 
			action = self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]):
			action = self._action_dict['right']
		elif np.array_equal(location, self._locations[2]):	
			action = self._action_dict['up']
		else:	# middle room
			assert np.array_equal(location, self._locations[1])
			# select the room to visit next uniformly at random
			r = random.random()
			# go 'left' or 'right' with the same likelyhood
			if (r < .5):
				action = self._action_dict['down']
			else:	
				action = self._action_dict['left']
		#self.logger.info("agt_prog: last loc {}, loc {}, dirty {}, action {}".\
		#			format(self._last_location, location, dirty, action))
		return action

	"""
	Random Reflex-based agent program for 'vacuum-3rooms-v0' world, 
	Check the map in 'maps.py'.
	"""	
	def agent_3rooms_v2(self, location, dirty):

		if dirty: 
			action = self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]):
			action = self._action_dict['right']
		elif np.array_equal(location, self._locations[2]):	
			action = self._action_dict['back']
		else:	# middle room
			assert np.array_equal(location, self._locations[1])
			# select the room to visit next uniformly at random
			random.seed(0)
			r = random.random()
			# go 'left' or 'right' with the same likelyhood
			if (r < .5):
				action = self._action_dict['right']
			else:	
				action = self._action_dict['left']
		#self.logger.info("agt_prog: last loc {}, loc {}, dirty {}, action {}".\
		#			format(self._last_location, location, dirty, action))
		return action
			
	"""
	Random Reflex-based agent program for 'vacuum-4rooms-v0' world.
	Nb: the agent program agent is the same as 'best-i-know' agent.
	Check the map in 'maps.py'. --__
	"""
	def agent_4rooms_v1(self, location, dirty):
		# the agent selects the next room randomly when it 
		# can't decide where to go
		if dirty: 
			action = self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]):
			action = self._action_dict['right']
		elif np.array_equal(location, self._locations[3]):	
			action = self._action_dict['left']
		elif np.array_equal(location, self._locations[1]):
			action = random.choice((self._action_dict['left'], self._action_dict['down']))
		else:	# location[2]
			action = random.choice((self._action_dict['right'], self._action_dict['up']))
		#self.logger.info("agt_prog: last loc {}, loc {}, dirty {}, action {}".\
		#			format(self._last_location, location, dirty, action))
		return action

	# 5 rooms map with + shape
	def agent_5rooms_v1(self, location, dirty):
		if dirty: 
			action = self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]):
			action = self._action_dict['down']
		elif np.array_equal(location, self._locations[1]):	
			action = self._action_dict['right']
		elif np.array_equal(location, self._locations[4]):	
			action = self._action_dict['up']
		elif np.array_equal(location, self._locations[3]):	
			action = self._action_dict['left']
		else: 
			#assert np.array_equal(location, self._locations[2])
			directions = ['left', 'down', 'right', 'up']
			str_action = random.choice(directions)
			action = self._action_dict[str_action]
		return action

	""" explore the square if rooms (3,4,1,0,3) counter-clock wise
		| |0|1|
		|2|3|4|
		|5| | | 
	"""
	def agent_6rooms_v1(self, location, dirty):
		if dirty: 
			action = self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]):
			stra = random.choice(('right', 'down'))
			action = self._action_dict[stra]
		elif np.array_equal(location, self._locations[1]):	
			stra = random.choice(('left', 'down'))
			action = self._action_dict[stra]
		elif np.array_equal(location, self._locations[2]):	
			stra = random.choice(('right', 'down'))
			action = self._action_dict[stra]
		elif np.array_equal(location, self._locations[3]):	
			stra = random.choice(('right', 'up', 'left'))
			action = self._action_dict[stra]
		elif np.array_equal(location, self._locations[4]):	
			stra = random.choice(('left', 'up'))
			action = self._action_dict[stra]
		else:
			#assert np.array_equal(location, self._locations[5])
			action = self._action_dict['up']
		return action
