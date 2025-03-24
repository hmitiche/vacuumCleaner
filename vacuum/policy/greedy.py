"""
Vacuum cleaner world, 2024.
'greedy.py'
author: Hakim Mitiche
date: April, 4th 2024
update: Nov 26th, 2024  
"""

from .base import CleanPolicy
from ..maps import Map
import numpy as np
import random
#import logging
#import os, os.path

""" 

"""
class GreedyPolicy(CleanPolicy):
	
	"""
	Base class for 'greedy' policy, for Vacuum Cleaner World.
	Usually such a policy uses a model such as last location
	or the number of visited rooms. The idea is to visit rooms 
	evenly (thus efficiently), to keep rooms clean as much as 
	possible, and, consequently, maximize the collected score.
	:param eco: economic mode flag.	It is for when the agent 
			knows dirt won't comeback to stop working after 
			the visit of all rooms. By default, we assume the 
			agent doesn't know about dirt re-appearance.
	"""
	def __init__(self, world_id, env, eco=False):
		super().__init__("greedy", world_id, env)
		# map locations ordred: left->right, up->down
		# @see: 'maps.py'
		self._locations = Map.locations_list(world_id)
		#print("world locations", self._locations)
		assert self._locations is not None
		self._env = env
		# does dirt comeback?
		self._dirt_comeback = env.unwrapped.dirt_comeback
		# to stop cleaning after visiting all rooms
		# even if dirt may comeback
		self._eco_mode = eco
		#env.get_wrapper_attr('_action_dict')
		# the number of rooms
		self._nbr_rooms = len(self._locations)
	

	def reset(self, seed=None):
		"""
		Reset variables used by the policy.
		call me after the end of a sim episode.
		add further variables needed by agent_program()
		"""
		self._visited = 0    			 # number of visited rooms
		self._last_location = None
		self._last_state_location = None
		# index of next direction to go  in some list 
		# (relevant to agent_5room_v1 for e.g, where list is a wheel)
		self._direction_index = 0	
		self._step_count = 0
		if not self._seeded or seed is not None:
			random.seed(seed)
			#print("[debug] random seeded with ", seed)
			self._seeded = True		

	
	def select_action(self, state):
		"""
		An implementation of CleanPolicy.select_action()
		Selects an action according to: current state, world map and policy
		"""
		room = state['agent']			# agent current location
		dirty = state['dirt']			# is there dirt in current room?
		# calls human defined policy
		action = self.agent_program(room, dirty)
		return action

	def agent_program(self, room, dirty):
		""" 
		'greedy' policy implemention for each map. 
		Typically, the agent is a reflex-based with model.
		
		:param: room agent current room
		:param: dust status in current room
		"""
		# update last location (utile often)
		self.update_world_model(room)
		# call the agent program according to the world map
		match (self.world_id):
			case "vacuum-2rooms":
				if self._location_sensor:
					act = self.agent_2rooms(room, dirty)
				else:
					act = self.agent_2rooms_blind(dirty)
			case "vacuum-3rooms-v0":
				act = self.agent_3rooms_v0(room, dirty)
			case "vacuum-3rooms-v1":
				act = self.agent_3rooms_v1(room, dirty)
			case "vacuum-3rooms-v2":
				act = self.agent_3rooms_v2(room, dirty)
			case "vacuum-4rooms-v0":
				act = self.agent_4rooms_v0(room, dirty)
			case "vacuum-4rooms-v1":
				act = self.agent_4rooms_v1(room, dirty)
			case "vacuum-4rooms-v2":
				act = self.agent_4rooms_v2(room, dirty)
			case "vacuum-5rooms-v0":
				act = self.agent_5rooms_v0(room, dirty)
			case "vacuum-5rooms-v1":
				act = self.agent_5rooms_v1(room, dirty)
			case "vacuum-5rooms-v2":
				act = self.agent_5rooms_v2(room, dirty)
			case "vacuum-5rooms-v3":
				act = self.agent_5rooms_v3(room, dirty)
			case "vacuum-5rooms-v4":
				act = self.agent_5rooms_v4(room, dirty)
			case "vacuum-6rooms-v0":
				act = self.agent_6rooms_v0(room, dirty)
			case "vacuum-6rooms-v1":
				act = self.agent_6rooms_v1(room, dirty)
			case "vacuum-6rooms-v2":
				act = self.agent_6rooms_v2(room, dirty)
			case "vacuum-7rooms-v0":
				act = self.agent_7rooms_v0(room, dirty)
			case "vacuum-7rooms-v1":
				act = self.agent_7rooms_v1(room, dirty)
			case "vacuum-8rooms-v0":
				act = self.agent_8rooms_v0(room, dirty)
			case "vacuum-8rooms-v1":
				act = self.agent_8rooms_v1(room, dirty)
			case "vacuum-9rooms-v0":
				act = self.agent_9rooms_v0(room, dirty)
			case _:
				act = None
				self.logger.critical("no agent program for ", self.world_id)
		
		# update last_state_location anyw	
		self._last_state_location = room
		self._step_count += 1
		return act


	def update_world_model(self, current_location):
		"""
		Updates the agent world model, 
		here it is merely the last room the vaccuum cleaner was in.
		
		:param: current_location (room)
		"""

		#self.logger.info("trying to update vacuum robot last location ...")
		#self.logger.info("location {}, last state location {}, last location {}".\
		#	format(current_location, self._last_state_location, self._last_location))
		# did movement succeed?
		if self._last_state_location is not None:
			if not np.array_equal(current_location, self._last_state_location):
				self._last_location = self._last_state_location
				# update the number of visited rooms
				self._visited += 1
				#self.logger.info("last location (updated): {}".format(self._last_location))	
			# otherwise, no change to last_location


	def agent_2rooms(self, location, dirty):
		"""
		a greedy agent program for 'vacuum-2rooms' world
		"""
		if dirty: 
			action = self._action_dict['suck']
		else:
			# stop if i know dirt won't comeback
			# and I have visited all rooms
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == 2:
					return self._action_dict['none']
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['right']
			elif np.array_equal(location, self._locations[1]):	
				action = self._action_dict['left']
		return action

	"""
	Greedy agent function for a vacuum cleaner robot without a location sensor, 
	in 'vacuum-2rooms' world.
	"""
	def agent_2rooms_blind(self, dirty):
		# is current room dirty?
		if dirty: 
			action = self._action_dict['suck']
		else:
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == 2:
					return self._action_dict['none']
			r = random.random()
			# go 'left' or 'right' with the same likelyhood
			if (r < .5):
				action = self._action_dict['right']
			else:	
				action = self._action_dict['left']
		return action


	def agent_3rooms_v0(self, location, dirty):
		"""
		Greedy agent for 'vacuum-3rooms-v0' world (map 1), 
		check the map in 'maps.py'.
		Agent type: reflex with model
		:param location: agent current room
		:param dirty: agent's current room state
		:return: an action in _action_dict
		"""	
		# is the current room dirty?
		if dirty: 	
			action = self._action_dict['suck']
		else:
			# stop if dirt doesn't comeback, and 
			# I've visited all the rooms (because movement is penalized)
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == 3:
					return self._action_dict['none']
			# the agent is in location 0 (and it's clean)
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['right']
			# the agent is in location 2 (and it's clean)
			elif np.array_equal(location, self._locations[2]):	
				action = self._action_dict['up']
			else:	# middle room (location 1)
				# avoid going back to the last visited room (self._last_location)
				if np.array_equal(self._last_location, self._locations[0]):
					action = self._action_dict['down']
				else: # last visited: locatio 2	
					action = self._action_dict['left']
		"""
		self.logger.info("agt_prog: last loc {}, loc {}, dirty {}, action {}".\
					format(self._last_location, location, dirty, action))
		"""
		# return the action to Gym simulator to simulate on the environment
		return action

	
	def agent_3rooms_v1(self, location, dirty):
		"""
		Greedy agent for 'vacuum-3rooms-v2' world, 
		check the map in 'maps.py'
		author: kezrane noor
		"""	
		# ida kant dirty
		if dirty: 	
			action = self._action_dict['suck']
		else:
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == 3:
					return self._action_dict['none'] #mandir walo
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['down']
			elif np.array_equal(location, self._locations[1]):	
				action = self._action_dict['right']
			else:
				if np.array_equal(self._last_location, self._locations[1]):
					action = self._action_dict['up']
				else:
					action = self._action_dict['left']
		return action

	
	#@FIXME: I should rather raise NotImplementedError
	def agent_3rooms_v2(self, location, dirty):
		"""
		greedy agent for 'vacuum-3rooms-v2' world, 
		check the map in 'maps.py'
		@nb: to be defined
		"""
		pass

	
	def agent_4rooms_v0(self, location, dirty):
		"""
		Greedy agent for 'vacuum-4rooms_v0' world,
		see map by typing: python main.py -v  (-|-)
		"""
		if dirty: 
			action = self._action_dict['suck']
		else:
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == self._nbr_rooms:
					return self._action_dict['none']
			# the agent visits the rooms in a clockwise rotation (an arbitrary choice)
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['right']
			elif np.array_equal(location, self._locations[1]):	
				action = self._action_dict['down']
			elif np.array_equal(location, self._locations[2]):	
				action = self._action_dict['up']
			else:
				action = self._action_dict['left']
			#self.logger.info("agt_prog: last loc {}, loc {}, dirty {}, action {}".\
			#			format(self._last_location, location, dirty, action))
		return action

	"""
	greedy agent for 4rooms_v1 world,
	see map in: testme.py, --__
	"""
	def agent_4rooms_v1(self, location, dirty):
		# the agent visits rooms in clockwise rotation
		# (just an arbitrary direction choice)
		if dirty: 
			action = self._action_dict['suck']
		else:
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == self._nbr_rooms:
					return self._action_dict['none'] 
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['right']
			elif np.array_equal(location, self._locations[1]):
				if np.array_equal(self._last_location, self._locations[0]):
					action = self._action_dict['down']
				else:
					action = self._action_dict['left']
			elif np.array_equal(location, self._locations[2]):	
				if np.array_equal(self._last_location, self._locations[1]):
					action = self._action_dict['right']
				else:
					action = self._action_dict['up']
			else:
				action = self._action_dict['left']
		#self.logger.info("agt_prog: last loc {}, loc {}, dirty {}, action {}".\
		#			format(self._last_location, location, dirty, action))
		return action

	"""
	The map looks like and an totally open-box when seen from above.
	Visit the rooms clock-wise, returning each time to the middle.
	The middle room is the most checked.
	"""
	def agent_5rooms_v1(self, location, dirty):
	
		# update the direction to go next when in the middle room
		# and only once when getting there 
		if np.array_equal(location, self._locations[2]) and\
		not np.array_equal(self._last_state_location, self._locations[2]):
			self._direction_index += 1 
		# the agent clean if dirty, 
		# the agent visits the rooms in a clockwise rotation
		# each time transiting by the middle room (location 2)
		if dirty: 
			action = self._action_dict['suck']
		else:
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == self._nbr_rooms:
					return self._action_dict['none']
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['down']
			elif np.array_equal(location, self._locations[1]):	
				action = self._action_dict['right']
			elif np.array_equal(location, self._locations[3]):	
				action = self._action_dict['left']
			elif np.array_equal(location, self._locations[4]):	
				action = self._action_dict['up']
			else:
				# room 2
				directions_weel = ['up','right','down','left']
				# start by going 'right', then rotate clock-wise
				# reset(): self._direction_index = 0  
				self._direction_index = self._direction_index%len(directions_weel) 
				goto = directions_weel[self._direction_index]
				action = self._action_dict[goto]
	
		return action

	def agent_5rooms_v4(self, location, dirty):
	
		# update the direction to go next when in the middle room
		# and only once when getting there 
		if np.array_equal(location, self._locations[2]) and\
		not np.array_equal(self._last_state_location, self._locations[2]):
			self._direction_index += 1 
		# the agent cleans if room is dirty, 
		# the agent visits rooms 0,1,3,4 in a clockwise rotation
		# then visits room room 2 from room 3
		if dirty: 
			action = self._action_dict['suck']
		else:
			# stops if all rooms are visited and rooms won't comeback
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == self._nbr_rooms:
					return self._action_dict['none']
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['right']
			elif np.array_equal(location, self._locations[1]):	
				action = self._action_dict['down']
			elif np.array_equal(location, self._locations[2]):	
				action = self._action_dict['right']
			elif np.array_equal(location, self._locations[4]):	
				action = self._action_dict['left']
			else:
				# room 3
				if np.array_equal(self._last_location, self._locations[2]):
					action = self._action_dict["up"]
				else:
					action = self._action_dict["left"]
		return action

	
	"""
	Greedy agent for six rooms version 0 map 
	"""
	def agent_6rooms_v0(self, location, dirty):
		if dirty: 
			action = self._action_dict['suck']
		else:
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == self._nbr_rooms:
					return self._action_dict['none']
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['right']
			elif np.array_equal(location, self._locations[1]):	
				action = self._action_dict['down']
			elif np.array_equal(location, self._locations[2]):	
				action = self._action_dict['up']
			elif np.array_equal(location, self._locations[3]):	
				action = self._action_dict['down']
			elif np.array_equal(location, self._locations[4]):	
				action = self._action_dict['up']
			elif np.array_equal(location, self._locations[5]):	
				action = self._action_dict['left']
			return action

	""" Explore the rooms that forms a square (3,4,1,0,3) 
		counter-clock wise, then go to (2,5) and loop.
		| |0|1|
		|2|3|4|
		|5| | | 
	"""
	def agent_6rooms_v1(self, location, dirty):
		if dirty: 
			action = self._action_dict['suck']
		else: 
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == self._nbr_rooms:
					return self._action_dict['none']
			if np.array_equal(location, self._locations[5]):
				action = self._action_dict['up']
			elif np.array_equal(location, self._locations[2]):
				if np.array_equal(self._last_location, self._locations[5]):	
					action = self._action_dict['right']
				else:
					action = self._action_dict['down']
			elif np.array_equal(location, self._locations[3]):	
				if np.array_equal(self._last_location, self._locations[0]):	
					action = self._action_dict['left']
				else:
					#assert np.array_equal(self._last_location, self._locations[2])\
					# or self._last_location is None
					action = self._action_dict['right']
			elif np.array_equal(location, self._locations[4]):	
					# always go counter-clockwise
				action = self._action_dict['up']
			elif np.array_equal(location, self._locations[1]):	
				action = self._action_dict['left']
			else: 
				assert np.array_equal(location, self._locations[0]) 
				action = self._action_dict['down']
		return action

	
	""" Explores the borders (0,1,2,5, ..3) in sequence 
		clock wise, then check room 4 before 
		closing the loop (eg. when the robot starts at 
		room 1, it visits then 2,5,8, .., 3, then 4, 
		after it comebacks to 3 and goes to room 0).
		|0|1|2|
		|3|4|5|
		|6|7|8| 
		@NB: check me, I maybe wron!
	"""
	def agent_9rooms_v0(self, location, dirty):
		if dirty: 
			action = self._action_dict['suck']
		else: 
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == self._nbr_rooms:
					return self._action_dict['none']
			if np.array_equal(location, self._locations[0]) or \
			np.array_equal(location, self._locations[1]):
				action = self._action_dict['right']
			elif np.array_equal(location, self._locations[2]) or \
			np.array_equal(location, self._locations[5]):
				action = self._action_dict['down']
			elif np.array_equal(location, self._locations[8]) or \
			np.array_equal(location, self._locations[7]) or \
			np.array_equal(location, self._locations[4]):
				action = self._action_dict['left']
			elif np.array_equal(location, self._locations[6]):
				action = self._action_dict['up']
			else: # room 3
				if np.array_equal(self._last_location, self._locations[6]):
					action = self._action_dict['right']
				else:
					assert np.array_equal(self._last_location, self._locations[4])
					action = self._action_dict['up']
		return action

		def agent_9rooms_v1(self, location, dirty):
			if dirty: 
				action = self._action_dict['suck']
			else: 
				if self._eco_mode and not self._dirt_comeback:
					if self._visited == self._nbr_rooms:
						return self._action_dict['none']
				if np.array_equal(location, self._locations[0]) or \
				np.array_equal(location, self._locations[1]):
					action = self._action_dict['right']
				elif np.array_equal(location, self._locations[2]) or \
				np.array_equal(location, self._locations[5]):
					action = self._action_dict['down']
				elif np.array_equal(location, self._locations[8]) or \
				np.array_equal(location, self._locations[7]) or \
				np.array_equal(location, self._locations[4]):
					action = self._action_dict['left']
				elif np.array_equal(location, self._locations[6]):
					action = self._action_dict['up']
				else: # room 3
					if np.array_equal(self._last_location, self._locations[6]):
						action = self._action_dict['right']
					else:
						assert np.array_equal(self._last_location, self._locations[4])
						action = self._action_dict['up']
			return action

