# Vacuum Cleaner World
# Gym environment for my AI course:
# chapter 2: intelligent agents
# to test agent programs written at classroom
# and RL programs
# Hakim Mitiche
# March 2024

# @fixme: the code needs re-organization, optimization and some cleaning
# @add: read env configuration from file: config.txt

__author__ = "Hakim Mitiche"
__email__ = "h.mitiche@gmail.com"
__status__ = "beta-version"

from world import VacuumCleanerWorldEnv
from vacuum import tools
from maps import Map
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import logging
import os, os.path
import sys

EPISODES = 1                  # default value
STEPS = 100                   # default value
SEED = 0                      # for env dynamics replication
LOG_PATH = 'log/'

# command-line syntax: 
# python testme.py [-h] map policy [-e episodes max_episode_steps] [-ls] [-cl]

def main():
	world_ids = Map.get_world_ids()		# from 'maps.py'
	policies = tools.get_policies()
	py_help = f"""
	Usage:  
		python testme.py [-h] map policy [-e episodes max_episode_steps] [-ls] [-cl]
			Runs the vaccum cleaner gym environment.
			map: 	vacuum world map number (0 ... {len(world_ids)-1})
			policy:	cleaning strategy number (0 ... {len(policies)-1})
			-e	set 'episodes' and 'max_episode_steps'
			episodes: number of simulation episodes (default: 1)
			max_episode_steps: for how many steps the MDP is ran? (default: 100)
			-ls set location sensor as faulty or abscent (default: working), 
			-cl clean flag. if set, the rooms remain clean after dusting (default: False)
			-h: display this Help.
		python testme.py -v [map_nbr]
			displays, on terminal console, the pre-defined vaccum world maps or 
			some given map.
			map_nbr:	the id number of the map to display 
	Examples:  
		'python testme.py 0 0 -e 5 1000' 
			simulates 'random' policy on 'vacuum-2rooms' map over 5 episodes
			each with 1000 steps, where the location sensor working correctly.
		'python testme.py 1 2 -ls'	
			simulates 'greedy' policy on 'vacuum-3rooms-v0' map over 1 
			episode with 100 steps and a location sensor that is faulty/abscent. 	
	"""
	args = sys.argv						# recuperate command-line args
	nbr_args = len(args)
	location_sensor = True              # the location sensor works (default)
	dirt_flag = True                    # dirt comeback after cleaning (default)
	policy_n = None						# policy id number
	# display script's help and examples
	if nbr_args==1 or nbr_args==2 and args[1] == '-h':
		print(py_help)
		# display available maps and policies
		addendum = "\t maps:   "
		ct = 0
		for i in world_ids:
			addendum += "{}: {}, ".format(ct, i)
			ct += 1
			if ct%3 == 0: addendum += "\n\t\t "
		addendum += "\n\n\t policies: "
		for i in policies.keys():
			addendum += "{}: {}, ".format(policies[i], i)
		print(addendum, "\n")
		exit()
	# display pre-defined world maps
	if args[1] == '-v':
		if nbr_args == 3:
			mapid = world_ids[int(args[2])]
			Map.display_map(mapid)
		else:
			Map.display_maps()
		exit()
	# parse command args
	try:
		# eg. python testme.py 1 1 -e 10 200 -ls -cl
		if (nbr_args >= 3):
			# make sure map and policy selection is correct
			if int(args[1]) not in range(len(world_ids)): 
				raise ValueError("map #{} doesn't exist!".format(args[1]))
			if int(args[2]) not in range(len(policies)):
				raise ValueError("policy #{} doesn't exist!".format(args[2]))
			world_id = world_ids[int(args[1])]
			world_map = Map.load_map(world_id)		# from 'maps.py'
			policy_n = int(args[2])
			for key in policies:
				if policies[key] == policy_n:
					policy_id = key
					break
			nbr_episodes = EPISODES    # default value
			max_steps = STEPS          # default value
		else:
			print("[error] you need to select a map and a policy (at least)!")
			print(py_help)
			exit()
		i = 3
		if nbr_args >= 6:
			if args[3] == '-e':
				nbr_episodes = int(args[4])
				max_steps = int(args[5])
				i = 6
		if i<nbr_args and args[i] == '-ls': 
			location_sensor = False
			i = i+1
		if i<nbr_args and args[i] == '-cl':
			dirt_flag = False
	except ValueError as error:
		print(repr(error))
		exit()
	mygym_name = "VacuumCleanerWorld-v0"
	print("Welcome to my custom Gymnasium environment: ", mygym_name)
	#print("[info] run log saved to file: {}".format(logfile))
	logfile = f"{LOG_PATH}{world_id}-{policy_id}.log"
	# delete the old logfile
	if os.path.isfile(logfile):
		os.remove(logfile)
		print("[info] logfile erased: {}".format(logfile))
	logging.basicConfig(filename=logfile, level=logging.DEBUG)
	logger = logging.getLogger(__name__)
	if (nbr_episodes > 1):
		#rmode = 'console'
		rmode = None
	else: rmode = 'human'
	# max_episode_steps: the duration the vaccum cleaner robot is powered on
	gym.register(
		 id="VacuumCleanerWorld-v0",
		 entry_point="world:VacuumCleanerWorldEnv",
		 #max_episode_steps=300,
		 # env args default values
		 kwargs={'grid':None, 'dirt_comeback':dirt_flag, 'dirt_proba':0.09, 'murphy_proba':0.11, 
		 'location_sensor':location_sensor, 'max_episode_steps':max_steps, 'render_mode':rmode}
	)
	# get vacuum world map represented as a 2D numpy's array
	wmap = np.array(world_map)
	# you may prefer 'console' render mode if there are many episodes
	# default is 'human'
	#env = gym.make('VacuumCleanerWorld-v0', grid=wmap, render_mode='console')
	env = gym.make('VacuumCleanerWorld-v0', grid=wmap)
	# seed the env, 
	# uncomment the next line for reproducible results
	# or for a single episode comparison
	env.reset(seed=SEED)
	env.unwrapped.set_map_name(world_id)		# for GUI randering
	logger.info(str(env.spec.kwargs))
	# add a wrapper to truncate simulation after max_episode_steps
	# TimeLimit unneeded, since I handle simulation termination condition in 
	# the env code using max_episode_steps
	#env = TimeLimit(env)		# maximum_episode_steps specified in register
	print("[info] Observation space: ", env.observation_space)
	print("[info] Action space: ", env.action_space)
	reward_dict = env.unwrapped.get_rewards()
	action_dict = env.unwrapped.get_actions()
	print("[info] Actions dictionary: ", action_dict)
	print("[info] Rewards dictionary: ", reward_dict)
	print('[info] Location sensor: ', location_sensor)
	eco = False
	# a greedy policy stops visiting rooms if it knows 
	# all have been cleaned and dirt won't comeback
	if not dirt_flag and policy_n in [1,2]:	# for greedy and greedy random agents
		answer = input("[prompt] does the agent knows that dirt won't comeback? [y/n]")
		if answer == 'y':	
			eco = True      # to be rational
	policy = tools.make_policy(policy_id, world_id, env, eco_mode=eco)
	policy.reset(seed=SEED)
	env.unwrapped.set_agent_name(policy_id)			# for GUI randering
	print("[info] Policy: {}".format(policy_id))
	print('[info] Map ID: {}'.format(world_id))
	print('[info] dirt comeback: ', dirt_flag)
	print('[info] simulating {} episodes...'.format(nbr_episodes))
	logger.info("[info] map:'{}'' policy:'{}' location_sensor:'{}' eps:{}\
		 max_steps:{}".format(world_id, policy_id, location_sensor,\
		 	nbr_episodes,max_steps))
	key = input("[prompt] press 'Enter' to start simulation!")
	if key!="": exit()
	# performance metrics per episode
	rewards = np.zeros(nbr_episodes)
	cleanings = np.zeros(nbr_episodes)
	travels = np.zeros(nbr_episodes)
	for eps in range(nbr_episodes):
		print("[info] episode {}, world configuration: ".format(eps+1))
		# reset the policy and the agent environment
		policy.reset()
		state = env.reset()[0]
		print("Initial state: (agent coordinates, dirt)=({},{}), world map: \n {}".\
			format(state['agent'], state['dirt'], wmap))
		print("step \t action  reward  state(ag_loc,dirt) \t info(dirty, act_done)")		
		done = False
		# NB: rendering, if any, is done before printing (below) action, reward, obs, ...
		while not done:
			action = policy.select_action(state)
			state, reward, done, truncated, info = env.step(action)
			step = info['step']
			dirty_rooms = info['dirty_spots']
			action_success = info['action_success']
			state_tuple = (state['agent'][0], state['agent'][1]) ,\
			 'dirty' if state['dirt'] else 'clean' 
			print(step, ": \t", action_dict[action], "\t", round(reward,2), "\t",\
			state_tuple, "\t (", dirty_rooms,",", action_success,")")
			if truncated: break

		#clean = env.get_wrapper_attr('_clean_rooms')
		rooms = env.get_wrapper_attr('_nbr_rooms')
		clean = rooms - dirty_rooms
		cleaned = env.get_wrapper_attr('_total_cleaned')
		messed = env.get_wrapper_attr('_total_messed')
		travel = env.get_wrapper_attr('_total_travel')
		reward = round(env.get_wrapper_attr('_episode_reward'), 2)
		rewards[eps] = reward
		cleanings[eps] = cleaned
		travels[eps] = travel
		if (rooms == clean):
			print("[info] mission accomplished! all rooms clean")
		else:
			print("[info] mission failed! dirty rooms left")
		# prnt stats to console
		print("[info] number of clean rooms (at the end): {}/{}".format(clean,rooms))
		print("[info] cleaned: {}, messed: {}".format(cleaned, messed))
		#logger.info("espisode {} reward {}".format(eps, reward))
		print("[info] total reward: ", reward)
		print('[info] total travels: ', travel)
	
	print("\n[info] simulation done!")
	print("[info] avg. reward per episode: ", round(sum(rewards)/nbr_episodes, 2))
	print("[info] avg. cleaning per episode: ", round(sum(cleanings)/nbr_episodes, 1))
	print("[info] avg. travels per episode: ", round(sum(travels)/nbr_episodes, 1))
	# invite the user to close pygame rendering
	if env.spec.kwargs['render_mode'] == 'human':
		input("[prompt] Press any key to close graphic rendering?")
	env.close()
	# only save results, if simulation is ran for a fair amount of episodes
	if nbr_episodes<50:
		exit()
	if input("[prompt] save results? [y/n]") == 'y':
		results = {'reward':rewards, 'cleaned':cleanings, 'travel':travels}
		tools.save_results(world_id, policy_id, results)
	print('[info] to plot results, type: ')
	print('[info] python vacuum.py -v [world_id]')

main()