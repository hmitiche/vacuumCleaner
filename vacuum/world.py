 
"""
world.py
________
Vaccum Cleaner World.
RL environment based on open AI Gymnasium to 
simulate intelligent agents doing vacuum cleaning.
The problem is described in my AI class for CS 
undergraduates, here (the link): 

author: Hakim Mitiche
date: March 2024
update: March 2025
"""

# @FIXME: use self.np_random in every rng!,
# add agent type, map name, in the upbar (to add too),
# remove debug assertions

from vacuum.maps import Map
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame
import numpy as np
import collections
import logging
import time

GYM_NAME = "VacuumCleanerWorld-v0"
LOG_FILENAME = 'logfile.log'

# Murphy law: "anything that can go wrong will go wrong
# anything that can't go wrong will go wrong".
# We applied the law for all vacuum actions, except 'none'

WRONG_PROBA = 0.17		# default value for 'self.murphy_proba'
INIT_DIRT_PROBA = 0.5	# probability that a room is dirty initally

# RGB colors 
LIGHT_BROWN = (225, 193, 110)
LIGHT_GREY = (211, 211, 211)
BLUE = (0, 0, 255)
BLUE2 = (51, 51, 255)
RENDER_FPS = 2	# rendering framerate (default value: 2 fps)
WINDOW_SIZE = 512 			# Default size of the PyGame window

class VacuumCleanerWorldEnv(gym.Env):
	
	""" VacuumCleanerWorld class, a custom Gymnasium environment
	for the vacuum cleaner world. The basic problem is  
	introduced by Russel & Norvig. 'AI: a Modern Approach' book 
	(Chapter 2: intelligent agents).  

	Attributes:
	-----------
	private:	
		map : np.array() dtype=int16
			vacuum cleaner world square map of size 'size', 4 * 4 at most.
		dirt_comback : boolean
			does dirt comeback to rooms?
		dirt_proba : float [0,1)
			probability of dirt coming back
		murphy_proba : float [0, 1]
			probability that an action fails or goes wrong.
			e.g., 'right' action may fail to take the agent the next room on the right, 
			'suck' may rather throw dirt and mess a clean room.
	public:
		observation_space : spaces.Box, spaces.discrete
			agent locations and dirt presence at current location

	Methods:
	--------	
	private:
		
	public:

	"""
	# render_fps: number of Frames Per Second in Pygame display
	metadata = {"render_modes": ["human", "console"], "render_fps": RENDER_FPS }

	def __init__(self, grid, dirt_comeback, dirt_proba, murphy_proba,\
		location_sensor, episode_max_steps, render_mode):
		super(VacuumCleanerWorldEnv, self).__init__()
		self.map = None					# world map current configuration (matrix)
		self.init_map = grid    		# world initial configuration
		self.map_name = "house"			# default map name
		self.agent_name = "vaccum cleaner robot"
		self.map_size = (grid.shape)[0] 	# the map (square) 1st dimension 
		self.dirt_comeback = dirt_comeback	# flag: does dirt reappear after cleaning?
		# the likelyhood of dirt reappearance after simulation step 
		self.dirt_proba = dirt_proba		
		self.murphy_proba = murphy_proba	# action failure probability
		# to simulate a partially observable environment, 
		# the robot location sensor maybe missing/not working
		self.location_sensor = location_sensor
		self.episode_max_steps = episode_max_steps	# to truncate simulation (if necessary)
		self.render_mode = render_mode
		self.nbr_rooms = self.count_rooms()
		self._episode = None		# current episode number
		self._step = None			# current episode step number
		self._episode_reward = None		# reward collected during the episode
		self._action_dict = self.get_actions()
		self._current_action = None
		self._action_success = None		# flag for whether the action is sucessfull
		# 'suck' effect on current room: 'cleaned', 'messed' or 'nothing'
		self._suck_outcome = None	
		# defines the observation space: a dictionary of the agent's locations, 
		# and dirt flag and number of visits to current room. 
		# The location is encoded as an element 
		# of {0, ..., `map_size-1`}^2, i.e. MultiDiscrete([size, size]).
		# dirt is encoded as 1 (present) or 0 (abscent)
		# likewise, the enrivonment is partialy observable
		self.observation_space = spaces.Dict(
			{
				"agent": spaces.Box(0, self.map_size, shape=(2,), dtype=np.int16),
				"dirt": spaces.Discrete(2),
				#"visits": spaces.Discrete(self.max_visits)
			}
		)

		# the vacuum cleaner robot can execute or receive 6 action orders:
		# "none", "suck", "right", "up", "left", "down"
		# encoded, respectively, as: 0,1,2,3,4,5
		self.action_space = spaces.Discrete(6)

		"""
		a dictionary that maps a travel action number (from `self.action_space`) 
		to 'décalage' with regard to X and Y-axis in the grid map.
		"""
		self._action_to_direction = {
			2: np.array([0, 1], dtype=np.int16),		# down
			3: np.array([1, 0], dtype=np.int16),		# right
			4: np.array([0, -1], dtype=np.int16),		# up
			5: np.array([-1, 0], dtype=np.int16),		# left
		}

		# make sure render_mode is set correctly, if set at all
		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode

		"""
		If we human-rendering is, `self.window` would be a reference
		to a Pygame window where we draw. 
		"""
		self.window_size = WINDOW_SIZE
		self.window = None
		"""
		the clock used to set framerate for human-mode rendering.
		defaut value, abscant.
		"""
		self.clock = None
		# initialize the logger object, uncomment if necessary
		logging.basicConfig(filename = LOG_FILENAME, level=logging.DEBUG)
		self.logger = logging.getLogger(__name__)

	def set_agent_name(self, name):
		# set the agent name
		self.agent_name = name

	def set_map_name(self, name):
		"""
		Sets the map name, which a unique identifier.
		Parameters:
			name (str): map name
		"""
		self.map_name = name

	def set_frame_rate(self, fps):
		"""
	    Set the frame rate (frames per second) for PyGame rendering.

	    Parameters:
	        fps (int): The desired number of frames per second.

	    This updates the 'render_fps' field in the environment's metadata, 
	    which can be used to control the update speed of visual rendering.
	    """
		metadata["render_fps"] = fps

	def _get_obs(self):
		"""
		Returns what the agent currently observes in the environment.
		Namely, where the agent is currently in and 
		the room state (dirty/clean)
		"""
		l = self._agent_location
		# sense dirt in the current room (x,y)=(l_1,l_0)
		if self.map[l[1],l[0]] == 'x': dirty = True
		else:  dirty = False
		self.logger.info("obs: {},{}".format(self._agent_location,dirty))
		return {"agent":self._agent_location, "dirt":dirty}
	  
		
	def _get_info(self):
		"""
		Gives further information on the environment after a simulation step.
		Returns: action success flag, number of dirty rooms left and step number 
		as a dictionary.
		"""
		nbr_dirty = self.count_rooms(clean=False)
		return{
			'action_success': self._action_success,
			'dirty_spots': nbr_dirty, 	# should not be available to the agent?!
			'step': self._step, 
		}

	def get_rewards(self):
		"""
		Defines how the environment reward the agent's actions.
		At each time step, the env. slightly rewards for a room that remains clean
		and evenly penalizes a dirty room, to encourage swift cleaning and exploration.
		A big reward is given for cleaning a dirty room. The env. discourages somehow 
		noise and power consumption by penalizing movement and sucking, but very slightly
		since the robot may need many of these. Furthermore, this helps the agent 
		to take a break when the dirt doesn't repear. Sucking dirt may fail (Murphy law) 
		and the robot may throw dirt and mess a clean room, this is penalized. 
		The rewards are deterministic.
		"""
		return{
			'clean': .1,		# reward for each clean room (per time unit)
			'cleaned': 10.,		# reward for cleaning a dirty room
			'dirty': -2.,		# penalty for a dirty room (per time unit)
			'suck': -.5,		# penalty of noise/power consumption
			'move': -.2,		# penalty of noise/power consumption
								# and to encourage shorter paths 
								# relevant when learning
			'throw': -5.,		# penalty for throwing dirt (murphy law) 
			'none': 0.,			# reward for being idle
		}

	def get_actions(self):
		"""
		Maps actions (represented as numbers) to their literal meaning.
		Returns: a dictionary, where key=action_id, value=action_name
		"""
		return{
			0:"none", 1:"suck", 
			2:"down", 3:"right", 4:"up", 5:"left"
		}

	def get_episode_reward(self):
		"""
		Returns the reward collected at the end of the episode
		"""	
		return self._episode_reward
   
	def sample_location(self):
		""" 
		Samples a map location that is correct (i.e:, 
		within the map boundries and not a black room '#')
		Usage: to generate the agent initial location or where 
		dirt may appear/reappear.
		:return: a location as an np.array(2)
		"""
		while True:
			# location is an np.array(): 
			# location[0]: X-coordinate, location[1]: Y-coordinate
			location = self.np_random.integers(0, self.map_size, size=2, dtype=np.int16)
			# check if the location is valid (not a black room)
			# the map coordinates are inverted when used as matrix indices
			x,y = location
			if self.map[y,x] != '#':
				break
		return location

	def sample_dirt(self, proba=None):
		""" 
		Simulates dirt re-appearance in clean rooms according to 'dirt_proba'. 
		To call only if 'dirt_comeback' flag is set to True. 
		Can be called during env.reset() to initially distribute dirt over 
		the map (such initial distribution may ve not been pre-defined in 'maps.py').
		:param proba: probability of dirt re-appearance at any room. 
		"""
		#assert self.dirt_comeback is True
		if proba is None:
			proba = self.dirt_proba
		row, col = self.map.shape
		for i in range(row):
			for j in range(col):
				# skip irrelevant rooms (already dirty and black rooms)
				if self.map[i,j] != '.': continue
				p = self.np_random.random()
				if (p < proba):
					self.map[i,j] = 'x'			# dirt appears in room (j,i)
					self._total_dirty += 1 
					#self.logger.info("dirt appeared in room {}".format((j,i)))
	
	def simulate_action(self, action):
		"""
		Simulates the success or failure of the robot's action (eg. 'Suck'). 
		According to Murphy's law, something wrong may happens, as when:
			1- an attempt to suck dirt fails
			2- sucking a clean room rather makes it dirty
			3- moving toward some direction fails, the robot stays in place.
		:return: True if the action succeed, False otherwise.
		"""
		# check if failure proba is set
		if self.murphy_proba is not None:
			p = self.np_random.random()
			if (p < self.murphy_proba): 
				self.logger.info(f"action '{self._action_dict[action]}' failed!")
				self._failures += 1
				return False
			self.logger.info("action '{}' succeeded!".format(self._action_dict[action]))
		return True

	def count_rooms(self, clean=None):
		""" 
		Counts specific rooms in the map (clean, dirty or all).
		:param clean (boolean): a flag to indicate whether to count clean, dirty 
				oe all rooms (default).
		:return: the number of rooms 
		"""
		#assert self.map is not None
		# a counter dict of rooms: clean, dirty and black rooms
		world = self.map if self.map is not None else self.init_map
		counters = collections.Counter(world.flatten())
		if clean == None:
			# number of clean room + number of dirty ones
			return (counters['.'] + counters['x'])
		if clean == True:
			return counters['.']		# clean 
		else:
			return counters['x']		# dirty

	
	def reset(self, seed=None, options=None):
		"""
		Resets the environment to some initial state
		:params: random number generator seed, 
			option is unused and i don't know what 
			is it?
		"""
		#assert self.logger is not None
		# seed self.np_random, the random number generator
		super().reset(seed=seed)
		if self._episode == None: 
			self._episode = 0
		else: 
			self._episode += 1 
		self._step = 0
		self._episode_reward = 0
		self.map = self.init_map
		self._total_dirty = 0
		# just in case all rooms are clean initially
		self.sample_dirt(proba=INIT_DIRT_PROBA)			
		self._total_cleaned = 0
		self._total_messed = 0              # nbr of rooms messed by the robot
		self._total_travel = 0              # total distance traveled by the vacuum robot
		self._failures = 0					# number of failures (Murphy law)
		self.nbr_rooms = rooms = self.count_rooms()
		clean = self.count_rooms(clean=True)
		#dirty = self.count_rooms(clean=False)
		dirty = rooms - clean
		self._total_dirty = dirty
		self.logger.info("env reset:")
		self.logger.info("rooms {}, clean {}, dirty {}".format(rooms, clean, dirty))
		# pickup the agent's start location uniformly at random
		self._agent_location = self.sample_location()
		self.logger.info("agent start location {}".format(self._agent_location))
		observation = self._get_obs()
		info = self._get_info()
		if self.render_mode == "human":
			self._render_frame()
		elif self.render_mode == "console":
			self._render_console()
		return observation, info
  
	def step(self, action):
		"""
		The implementation of 'step' method. 
		It simulate the agent action on the environment.
		:param action: the action that the agent just executed.
		:return: a new state, reward, observation, terminated, truncated, info 
		"""
		self._current_action = action
		self._suck_outcome = None
		reward = 0
		# suck dirt if any
		if action == 0:
			#self.logger.info("action: 'none'")
			reward = self.get_rewards()['none']
			self._action_success = True
			# stay still
		elif action == 1:	# 'suck'
			self.logger.info("action: 'suck'")
			# penality for noise and power consumption
			penalty = self.get_rewards()['suck']
			x, y = self._agent_location
			# an agent is either in a clean pr dirty room
			if self.map[y,x] == 'x':
				#self.logger.info("agent room {} 'dirty'".format((x,y)))
				# to simulate a variable cleaning time  
				# and 'cleaning' failure
				# (see WRONG_PROBA above)
				self._action_success = self.simulate_action(action)
				if self._action_success:
					self.map[y,x] = '.'		# room becomes clean
					self._suck_outcome = 'cleaned'
					self.logger.info("room {} is cleaned!".format((x,y)))
					self._total_cleaned += 1
					# add a reward for cleaning
					reward = penalty + self.get_rewards()['cleaned']	
				else:
					# otherwise (failure), room remains dirty 
					self._suck_outcome = 'nothing'
					self.logger.info("room {} remains dirty!".format((x,y)))
			else:
				# simulate 'suck' action throwing dirt in a clean room
				assert self.map[y,x] == '.'			# a room either clean or dirty
				self.logger.info("agent dusting clean room ".format((x,y)))
				self._action_success = self.simulate_action(action)
				# murphy law: 'suck' may mess a clean room
				if not self._action_success:
					self.map[y,x] = 'x'
					self._total_dirty += 1
					self._total_messed += 1
					self._suck_outcome = 'messed'
					self.logger.info("agent throws dirt in clean room {}!".format((x,y)))
					reward = penalty + self.get_rewards()['throw']
				else:
					self._suck_outcome = 'nothing'
		else:
			assert action in (2,3,4,5)		# movement actions
			self.logger.info("action: '{}'".format(self._action_dict[action]))
			# penalize movement even if it doesn't result in a change of room
			reward = self.get_rewards()['move']
			self._action_success = self.simulate_action(action)
			if self._action_success:
				self._total_travel += 1
				# map the move action ({2,3,4,5}) to the direction
				direction = self._action_to_direction[action]
				# get new location, use `np.clip` to stay within the grid
				new_location = np.clip(
					self._agent_location + direction, 0, self.map_size - 1
				)
				a,b = self._agent_location
				c,d = new_location
				# check for movement, obstacles(boundry) or black rooms
				if not np.array_equal(self._agent_location, new_location):
					# new location must be not a black room
					if self.map[new_location[1], new_location[0]] != '#':
						self._agent_location = new_location
						self._action_success = True
						self.logger.info("agent moved to ({},{})".format(c,d))
					else:
						self._action_success = False
						self.logger.info("can't move to ({},{})! black room".format(c,d))
				else:
					# stay otherwise in current location (map boundry)
					self._action_success = False
					self.logger.info("can't move out of boundry, staying in {}"\
						.format((c,d), (a,b)))
			else:
				self.logger.info("movement failed! (Murphy law) :(")	 
		
		terminated = False
		self._step += 1
		# make sure an episode does exceed the limit T=episode_max_steps
		truncated = (self._step == self.episode_max_steps)
		# compute step reward
		nbr_clean = self.count_rooms(clean=True)
		nbr_dirty = self.count_rooms(clean=False)
		#assert nbr_clean is not None
		reward = reward + nbr_clean * self.get_rewards()['clean']\
		+ nbr_dirty * self.get_rewards()['dirty']
		self._episode_reward = round(self._episode_reward + reward, 2)

		# update dirt distribution if necessary
		if self.dirt_comeback:
			self.sample_dirt()
		observation = self._get_obs()
		info = self._get_info()

		if self.render_mode == "human":
			self._render_frame()
		elif self.render_mode == "console":
			self._render_console()
		# no rendering otherwise

		return observation, reward, terminated, truncated, info

	def render(self):
		"""
		Show simulation in console or GUI
		"""
		#assert self.render_mode is not None	
		if self.render_mode == "console":
			self._render_console()
		elif self.render_mode == "human":
			return self._render_frame()

	def _render_console(self):
		"""
		Displays simulation to Terminal console
		"""
		l = self._agent_location
		coord = l[0], l[1]
		#print("agent's room:{}, world map:\n {}".format(coord, self.map))
		print("agent room: ({})".format(coord))
		print("world map: \n ", self.map)

	# @NB: you did a great job
	def _render_frame(self):
		"""
		This renders vacuum cleaner robot and environment for humans by using PyGame.
		"""
		if self.window is None:
			pygame.init()
			pygame.display.init()
			self.upbar_size = 70
			self.downbar_size = 110
			dim_x, dim_y = self.window_size, (self.window_size+self.upbar_size+self.downbar_size)
			dimension = (dim_x,dim_y)
			self.window = pygame.display.set_mode(dimension)
			pygame.display.set_caption("Vacuum Cleaner World-v0 (OpenAI Gym)")
		if self.clock is None:
			self.clock = pygame.time.Clock()
		# map pygame surface
		canvas = pygame.Surface((self.window_size, self.window_size))
		# fill screen with clean rooms
		bg_color = (255, 255, 255)			# 'while' background color
		canvas.fill(bg_color)
		self.window.fill(bg_color)
		font0_name = "Arial"
		font1_name = "comicsans"
		font2_name = "Helvetica"
		font3_name = 'Starjedi.ttf'
		# display agent name and cumulated reward
		upbar_text1 = 'agent'
		upbar_text2 = 'reward'
		upbar_text3 = f'{self.agent_name}'
		upbar_text4 = f'{self._episode_reward}'
		font = pygame.font.SysFont(font0_name, 18, True)
		text_surface1_1 = font.render(upbar_text1, True, "black")
		text_surface1_2 = font.render(upbar_text2, True, "black")
		font = pygame.font.SysFont(font0_name, 27, False)
		text_surface1_3 = font.render(upbar_text3, True, "black")
		text_surface1_4 = font.render(upbar_text4, True, "black")
		rect_1 = text_surface1_1.get_rect(topleft=(10,5))
		rect_2 = text_surface1_2.get_rect(topright=(self.window_size-10,5))
		rect_3 = text_surface1_3.get_rect(topleft=tuple(map(sum, zip(rect_1.bottomleft, (0,5)))))
		rect_4 = text_surface1_4.get_rect(topright=tuple(map(sum, zip(rect_2.bottomright, (0,5)))))
		#rect_1.right = rect_3.right = 50
		#rect_3.top = text_surface1_1.get_height()+10
		#rect_2.left = rect_4.left = text_surface1_2.get_width() + 50
		#rect_4.top = text_surface1_3.get_height()+10
		#if self._step == self.episode_max_steps:
		#	upbar_text += '  [end]'
		# display game stats
		stats_text_1 = 'Episode {}, Step {}:'.format(self._episode, self._step)
		stats_text_2 = f'Dirt {self._total_dirty}  Clean {self._total_cleaned}'\
		+f'  Mess {self._total_messed}  Travel {self._total_travel}  Fail {self._failures}'
		# display world name and configuration
		config_text_1 = f"Map '{self.map_name}':"
		if self.dirt_comeback:
			config_text_2 = f'dirt_comeback:{self.dirt_comeback}  P_dirt={self.dirt_proba}'
		else:
			config_text_2 = f"dirt_comeback:{self.dirt_comeback}"
		config_text_2 += f'  P_wrong={self.murphy_proba}  loc_sensor:{self.location_sensor}'
		#font_police = 'coffeehouse.ttf'
		#font = pygame.font.Font(font_police, 18)
		#text_surface1.fill("black")
		#font = pygame.font.Font('freesansbold.ttf', 14)
		font = pygame.font.SysFont(font0_name, 16, True)
		text_surface2_1 = font.render(stats_text_1, True, "black")
		text_surface3_1 = font.render(config_text_1, True, "black")
		font = pygame.font.SysFont(font0_name, 18, False)
		text_surface2_2 = font.render(stats_text_2, True, "black")
		font = pygame.font.SysFont(font0_name, 16, False)
		text_surface3_2 = font.render(config_text_2, True, "black")
		#text_surface.get_rect().fill(bg_color)
		pix_square_size = self.window_size / self.map_size    # room (square) size (x*x)
		# draw dirty rooms as light grey rectangles
		dirty_color = LIGHT_BROWN
		# draw dirty rooms
		for dirty_room_location in self._map_locations(symbol='x'):
			pygame.draw.rect(
				canvas,
				dirty_color,
				pygame.Rect(
					pix_square_size * dirty_room_location,	# coordinates
					(pix_square_size, pix_square_size),
				),
			)

		# draw the vacuum cleaner (a blue circle with a triangle inside 
		# that indicates next direction to take)
		# current room center coordinates
		center = (self._agent_location + 0.5) * pix_square_size
		radius = pix_square_size / 4			# radius
		ag_color = BLUE2
		#ag_width = None
		if self._current_action == 1:			# 'suck'
			x,y = self._agent_location[1], self._agent_location[0]
			if self._suck_outcome == 'cleaned':
				#assert self.map[x,y] == '.'     # room cleaned (but dirt may just comeback)
				ag_color = LIGHT_BROWN
			elif self._suck_outcome == 'messed':
				assert self.map[x,y] == 'x'
				# change robot color to white to signal dirt throwing
				ag_color = "white"
			#else: 		
			#	assert self._suck_outcome == 'nothing'
			#	ag_color = "red"
		else: 
			if self._current_action != 0:	# nothing
				if self._action_success:	ag_color = BLUE2
				else:	ag_color = "red"
		pygame.draw.circle(
			canvas,
			ag_color, 	#,"green", #BLUE,
			center,
			radius,
			#ag_width,
		)
		# if it's a movement action, draw an triangle inside the circle
		# indicating the robot direction
		if self._current_action in [2,3,4,5]: 
			x, y = center[0], center[1]
			p1, p2, p3 = self.get_triangle_points(self._current_action, x, y, radius)
			pygame.draw.polygon(
				canvas,
				"yellow",
				(p1, p2, p3),
			)

		# draw black rooms		
		for black_room_location in self._map_locations(symbol='#'):
			pygame.draw.rect(
				canvas,
				(0, 0, 0),
				pygame.Rect(
					pix_square_size * black_room_location,
					(pix_square_size, pix_square_size),
				),
			)
		
		# draw gridlines
		for x in range(self.map_size + 1):
			pygame.draw.line(
				canvas,
				0,
				(0, pix_square_size * x),
				(self.window_size, pix_square_size * x),
				width=3,
			)
			pygame.draw.line(
				canvas,
				0,
				(pix_square_size * x, 0),
				(pix_square_size * x, self.window_size),
				width=3,
			)

		if self.render_mode == "human":
			#text_rect = text_surface1.get_rect(center=(self.window_size//2, self.upbar_size//2))
			#self.window.blit(text_surface1, text_rect)
			self.window.blit(text_surface1_1, rect_1)
			self.window.blit(text_surface1_2, rect_2)
			self.window.blit(text_surface1_3, rect_3)
			self.window.blit(text_surface1_4, rect_4)
			# copy drawings from `canvas` to the visible window
			#self.window.blit(canvas, canvas.get_rect())
			x = 10
			#x_margin = 160
			x_margin = 0
			self.window.blit(canvas, (0, self.upbar_size))
			y = self.window_size + self.upbar_size + 10
			self.window.blit(text_surface2_1, (x, y))
			y = y + text_surface2_1.get_height()+5
			self.window.blit(text_surface2_2, (x+x_margin, y))
			y = y + text_surface2_2.get_height()+10
			self.window.blit(text_surface3_1, (x, y))
			y = y + text_surface3_1.get_height()+5
			self.window.blit(text_surface3_2, (x+x_margin, y))
			pygame.event.pump()
			pygame.display.update()

			# delay rendering according to a framerate suitable for human.
			self.clock.tick(self.metadata["render_fps"])
			# check for the quit event
			for event in pygame.event.get():
				# user pressed close button
				if event.type == pygame.QUIT:
					pygame.display.quit()
					pygame.quit()
					exit()				# stop simulation				


	def get_triangle_points(self, action, x, y, r):
		"""
		Rotates the triangle inside the robot to reflect 
		the next direction the will take.
		"""
		match (action):
			case 2: p1, p2, p3 = (x,y+r),(x-r,y),(x+r,y)
			case 3: p1, p2, p3 = (x,y-r),(x,y+r),(x+r,y)
			case 4: p1, p2, p3 = (x-r,y),(x,y-r),(x+r,y)
			case 5: p1, p2, p3 = (x-r,y),(x,y+r),(x,y-r)
		return p1,p2,p3

	def _map_locations(self, symbol):
		"""
		Returns map locations ordered (left->right, up->down)
		as a list of 1D arrays (plan coordinates)
		:param symbol: designate the kind of room to consider 
			(x: dirty, .: clean)
		@redundant, see: locations_list in 'tools.py'
		"""
		l = []; dim = self.map_size
		for i in range(dim):
			for j in range(dim):
				if (self.map[i,j] == symbol):
					l.append(np.array([j,i], dtype=np.int16))
		return l

	def close(self):
		"""
		Terminates the simulation by releasing the resources created for rendering.
		"""
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()