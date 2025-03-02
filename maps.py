# 'maps.py'
# Vacuum Cleaner World
# Some maps I defined for the smart vaccum cleaner robot.
# Hakim Mitiche
# March 2024
import numpy as np
import pprint

# @fix_me: add initial dirt distribution in the maps, 
# in case dirt doesn't comeback, though
# you can do it in 'world.py' by calling sample_dirt()
# during env.reset()

class Map:

	# Vacuum world pre-defined maps IDs
	world_ids = ["vacuum-2rooms", 
	"vacuum-3rooms-v0", "vacuum-3rooms-v1", "vacuum-3rooms-v2", 
	"vacuum-4rooms-v0", "vacuum-4rooms-v1", "vacuum-4rooms-v2", "vacuum-4rooms-v3",
	"vacuum-5rooms-v0", "vacuum-5rooms-v1", "vacuum-5rooms-v2", "vacuum-5rooms-v3",
	"vacuum-5rooms-v4",
	"vacuum-6rooms-v0", "vacuum-6rooms-v1", "vacuum-6rooms-v2",
	"vacuum-7rooms-v0", 
	"vacuum-8rooms-v0", "vacuum-8rooms-v1",
	"vacuum-9rooms-v0"]

	world_maps = None           # list of pre-defined maps
	# currently selected world (to share with other '.py')
	world_map = None            # map data
	world_id = None             # map identifier


	"""
	Convenient display of vaccum map in console terminal.
	:param: matrix	square array corresponding to the map.
	"""
	@staticmethod
	def pretty_print(matrix):
		# Loop over each row
		for row in matrix:
			# Convert each element to a string and join with spaces
			print(' '.join(map(str, row)))
 
	"""
	display the vacuum world maps on the terminal console
	"""
	@classmethod
	def display_maps(cls):
		if cls.world_maps == None:
			map_list = cls.load_all_maps(Map)
		assert len(cls.world_ids)==len(map_list), "map and IDs lists lengths don't match!"
		#pp = pprint.PrettyPrinter(indent=1)
		for i,world in enumerate(cls.world_ids):
			#colum = f"{world}"
			print(f"Map {i}: '{world}'")
			Map.pretty_print(map_list[i])

	"""
	display a given vacuum world map on the terminal.
	"""
	@classmethod
	def display_map(cls, mapid):
		if cls.world_maps == None:
			map_list = cls.load_all_maps(Map)
		assert len(cls.world_ids)==len(map_list), "map and IDs lists lengths don't match!"
		#pp = pprint.PrettyPrinter(indent=0)
		print(f"Map '{mapid}':")
		Map.pretty_print(map_list[cls.get_map_index(Map, mapid)])

	@classmethod
	def get_world_ids(cls):
		return cls.world_ids

	"""
	Return an ordered list of room for a given world map (left2right, up2down)
	e.g. for 2nd case: map =[A,B,#  
							 #,C,#
							 #,#,#],
		 location_list= [A,B,C]
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
			raise ValueError(f"there is no map for {world_id}")
			exit()
		dim = len(wmap)
		loc_list = [np.array([x,y])
					for y in range(dim) 
					for x in range(dim)
					if wmap[y][x] != '#']
		return loc_list
	
	"""
	Return a map from vacuum world maps pool, given the map ID: mapid.
	The agent initial location is sampled during env.make()
	inside VacuumCleanerWorlEnv ('world.py')
	"""
	@classmethod
	def load_map(cls, mapid):
		assert mapid in cls.world_ids, f"there is no map: {cls.mapid}"
		world_maps = cls.load_all_maps(Map)
		map_ind = cls.get_map_index(Map, mapid)
		cls.world_id = mapid
		cls.world_map = world_maps[map_ind]
		return cls.world_map

	def get_map_index(cls, mapid):  
		return [i for i in range(len(cls.world_ids)) if cls.world_ids[i]==mapid][0]

	"""
	get the list of pre-defined vaccum cleaning maps
	""" 
	def load_all_maps(cls):
		"""
		define world maps along with initial dirt distributions.
		The grid must be a square, (eg. world_map2_0 is 3x3).
		'#': obstacle or a closed room (to ignore)
		'.': clear room
		'x': dirty room
		"""
		
		# 2 rooms map, form: --
		world_map2_0 = [
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

		world_maps = [
		world_map2_0, world_map3_0, world_map3_1, world_map3_2,
		world_map4_0, world_map4_1, world_map4_2, world_map4_3, 
		world_map5_0, world_map5_1, world_map5_2, world_map5_3, world_map5_4,
		world_map6_0, world_map6_1, world_map6_2, 
		world_map7_0, world_map8_0, world_map8_1, world_map9_0
		]

		return world_maps
