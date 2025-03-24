"""
'maps.py'
---------
Vacuum Cleaner World v0
A group of maps I defined for the smart vaccum cleaner robot.
Most are 3 by 3 grids where rooms maybe unaccessible 
(black rooms). The initial distribution of dirt is given, 
the robot depart location isn't. Many doesn't have dirt, 
they left to fill by env.reset()
Author: Hakim Mitiche
Date: March 2024
"""

import numpy as np
import pprint

class Map:


	"""
	NB: make sure there is an initial dirt distribution in the maps, 
	in case dirt doesn't comeback. To do so, the method 'sample_dirt()'
	must be called during env.reset() in in 'world.py'.
	"""
	# list of vacuum world maps IDs (defined here)
	# name: 'vacuum-[n]rooms-v[m]', n: number of rooms, m: map version 
	# number among maps with the same n
	world_ids = ["vacuum-2rooms", 
	"vacuum-3rooms-v0", "vacuum-3rooms-v1", "vacuum-3rooms-v2", 
	"vacuum-4rooms-v0", "vacuum-4rooms-v1", "vacuum-4rooms-v2", "vacuum-4rooms-v3",
	"vacuum-5rooms-v0", "vacuum-5rooms-v1", "vacuum-5rooms-v2", "vacuum-5rooms-v3",
	"vacuum-5rooms-v4",
	"vacuum-6rooms-v0", "vacuum-6rooms-v1", "vacuum-6rooms-v2",
	"vacuum-7rooms-v0", 
	"vacuum-8rooms-v0", "vacuum-8rooms-v1",
	"vacuum-9rooms-v0", 
	"vacuum-11rooms-v0"]

	world_maps = None           # list of pre-defined maps
	# currently selected world (to share with other '.py')
	world_id = None             # (selected) map identifier (alphanumeric String)
	world_map = None            # (selected) map data 

	"""
	a friendly display of a vacuum map in terminal console.
	:param matrix: a square shaped array that is associated to the map.
	"""
	@staticmethod
	def pretty_print(matrix):
		# Loop over each row
		for row in matrix:
			# Convert the elements to a strings which joined by spaces
			print(' '.join(map(str, row)))
 
	@classmethod
	def display_maps(cls):
		"""
		Displays (available) vacuum worlds on the terminal console
		"""
		if cls.world_maps == None:
			map_list = cls.load_all_maps(Map)
		assert len(cls.world_ids)==len(map_list), "map and IDs lists lengths don't match!"
		#pp = pprint.PrettyPrinter(indent=1)
		print("Legend: '.': clean room   'x': dirty room   '#': walls")
		for i,world in enumerate(cls.world_ids):
			#colum = f"{world}"
			print(f"Map {i}: '{world}'")
			Map.pretty_print(map_list[i])
			print("")

	"""
	Displays a given vacuum world map on the terminal.
	Parameters:
		mapid (String): the identifier of the map to display
	"""
	@classmethod
	def display_map(cls, mapid):
		if cls.world_maps == None:
			map_list = cls.load_all_maps(Map)
		assert len(cls.world_ids)==len(map_list), "map and IDs lists lengths don't match!"
		#pp = pprint.PrettyPrinter(indent=0)
		print("Legend: '.': clean room, 'x': dirty room, '#': walls")
		print(f"Map '{mapid}':")
		Map.pretty_print(map_list[cls.get_map_index(Map, mapid)])
		print("\n")

	@classmethod
	def get_world_ids(cls):
		return cls.world_ids

	"""
	Returns a sorted list of a map rooms, as (x,y) coordinates, in this order: 
	from left 2 right and from up 2 down.
	:param world_id: the map identifier
	:return: list of locations (rooms)

	Example:
		world_map3_0: map =[A,B,#  
							 #,C,#
							 #,#,#],
		 locations_list= [A,B,C]
	"""
	@classmethod
	def locations_list(cls, world_id):
		"""
		count = 0
		wmap = None
		for wid in cls.world_ids:
			if wid == world_id:
				wmap = cls.world_maps[count]
				break
		"""
		assert cls.world_id == world_id
		wmap = cls.world_map
		if wmap == None:
			raise ValueError(f"there is no map for '{world_id}'")
			exit()
		dim = len(wmap)
		loc_list = [np.array([x,y])
					for y in range(dim) 
					for x in range(dim)
					if wmap[y][x] != '#']
		return loc_list
	
	@classmethod
	def load_map(cls, mapid):
		"""
		Returns a map from vacuum world maps pool.
		:param mapid: the selected map ID
		Notes:
		The agent initial location is sampled during env.make()
		inside VacuumCleanerWorlEnv ('world.py')
		"""
		assert mapid in cls.world_ids, f"there is no map: {cls.mapid}"
		world_maps = cls.load_all_maps(Map)
		map_ind = cls.get_map_index(Map, mapid)
		cls.world_id = mapid
		cls.world_map = world_maps[map_ind]
		return cls.world_map

	def get_map_index(cls, mapid):  
		"""
		Returns the selected map index in the pre-defined list (cls.world_maps)

		"""
		return [i for i in range(len(cls.world_ids)) if cls.world_ids[i]==mapid][0]


	def load_all_maps(cls):
		"""
		Defines some maps along with the initial dirt distributions.
		They are grids of a square shape, mostly here 3x3. The user 
		can add further map here.  The grid values are:
		'#': obstacle or a closed room (to ignore)
		'.': clean room
		'x': dirty room

		Returns: the pre-defined list of vaccum cleaning maps
		"""
		
		# 2 rooms map, form: --
		world_map2 = [
			['x', 'x', '#'],
			['#', '#', '#'],
			['#', '#', '#'],
		]
		# 3 rooms v0, form: '|
		world_map3_0 = [
			['x', '.', '#'],
			['#', 'x', '#'],
			['#', '#', '#'],
		]
		# 3 rooms v1, form __|
		world_map3_1 = [
			['#', '.', '#'],
			['.', '.', '#'],
			['#', '#', '#'],
		]
		# 3 rooms v2, form: --- (rooms disposed in a straight line)
		world_map3_2 = [
			['.', '.', 'x'],
			['#', '#', '#'],
			['#', '#', '#'],
		]
		# 4 rooms v0, that froms a square (easy)
		world_map4_0 = [
			['x', '.', '#'],
			['x', 'x', '#'],
			['#', '#', '#'],
		]
		# 4 rooms v1, a form of tetris game block --__
		world_map4_1 = [
			['x', '.', '#'],
			['#', 'x', '.'],
			['#', '#', '#'],
		]
		# 4 rooms v2, form: __|__ 
		world_map4_2 = [
			['#', '.', '#'],
			['x', '.', 'x'],
			['#', '#', '#'],
		]
		# 4 rooms v3
		world_map4_3 = [
			['#', '#', 'x'],
			['.', 'x', '.'],
			['#', '#', '#'],
		]
		# 5 rooms v0, form: __i-^, 
		world_map5_0 = [
			['#', '#', 'x'],
			['#', '.', 'x'],
			['x', '.', '#'],
		]
		# 5 rooms v1, form : +
		world_map5_1 = [
			['#', '.', '#'],
			['.', 'x', '.'],
			['#', 'x', '#'],
		]
		# 5 rooms v2
		world_map5_2 = [
			['.', '.', '#'],
			['#', '.', '#'],
			['.', 'x', '#'],
		]
		# 5 rooms v3, form: 
		world_map5_3 = [
			['#', '#', '.'],
			['.', '.', '.'],
			['.', '#', '#'],
		]
		# 5 rooms v4, form: 
		world_map5_4 = [
			['#', '.', '.'],
			['.', '.', '.'],
			['#', '#', '#'],
		]
		# 6 rooms v0, easy one
		world_map6_0 = [
			['.', '.', '#'],
			['.', '.', '#'],
			['.', '.', '#'],
		]
		# 6 rooms v0, easy one
		world_map6_1 = [
			['#', '.', '.'],
			['.', '.', '.'],
			['.', '#', '#'],
		]
		# 6 rooms v0, easy one
		world_map6_2 = [
			['.', '#', '#'],
			['.', '.', '.'],
			['.', '#', '.'],
		]
		# 7 rooms v0
		world_map7_0 = [
			['.', '#', '.'],
			['.', '.', '.'],
			['.', '#', '.'],
		]

		# 8 rooms v0, form: a donut
		world_map8_0 = [
			['.', '.', '.'],
			['.', '#', '.'],
			['.', '.', '.'],
		]

		# 8 rooms v1, easy 
		world_map8_1 = [
			['#', '.', '.'],
			['.', '.', '.'],
			['.', '.', '.'],
		]

		# 9 rooms v0, full 3x3 map
		world_map9_0 = [
			['.', '.', '.'],
			['.', '.', '.'],
			['.', '.', '.'],
		]

		# 11 rooms v0, 4 x 4 grid
		world_map11_0 = [			# 
			['.', '.', '#','#'],
			['#', '.', '.', '#'],
			['.', '.', '.', '.'],
			['#', '.', '.', '.']
		]

		world_maps = [
		world_map2, 
		world_map3_0, world_map3_1, world_map3_2,
		world_map4_0, world_map4_1, world_map4_2, world_map4_3, 
		world_map5_0, world_map5_1, world_map5_2, world_map5_3, world_map5_4,
		world_map6_0, world_map6_1, world_map6_2, 
		world_map7_0, 
		world_map8_0, world_map8_1, 
		world_map9_0, 
		world_map11_0
		]

		return world_maps
