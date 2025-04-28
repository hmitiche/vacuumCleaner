"""
main.py
------- 
Vacuum Cleaner World
Gymansium environment for my AI course:
chapter 2: intelligent agents
to illustrate agent programs written at classroom
and RL programs

Author: Hakim Mitiche
Date: March 2024
Version: 1.0
Licence: none

Usage:
	python vacuumclean [-h] map policy [-e episodes episode_max_steps] [-ls] [-cl]

Notes:
	- Requires: gymnasium, matplotlib, pygame, numpy and pickle
	- You need to run this on terminal console

License:
	none

@fixme: the code needs re-organization, optimization and some cleaning
@added: read the env. configuration, filename: config.txt

__author__ = "Hakim Mitiche"
__email__ = "h.mitiche@gmail.com"
__status__ = "beta-version"
"""
from vacuum.policy.qlearning import QLearnPolicy
from vacuum.world import VacuumCleanerWorldEnv
from vacuum.tools import Tools
from vacuum.maps import Map
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import logging
import os, os.path
import sys

EPISODES = 1        # default value
STEPS = 100                   # default value
SEED = 0                      # RNG seed, for env dynamics replication
LOG_PATH = 'log/'

# main function
def main():
	args = sys.argv						# recuperate command-line args
	print_help()
	world_id, policy_id, nbr_episodes, max_steps, dirt_flag, \
	location_sensor = parse_commandline_args(args)
	world_map = Map.load_map(world_id)		# from 'maps.py'
	mygym_name = "VacuumCleanerWorld-v0"
	print(f"Welcome to {mygym_name}, a custom open AI Gymnasium environment!")
	#print("[info] simulation log saved to file: {}".format(logfilename))
	logfilename = f"{LOG_PATH}{world_id}-{policy_id}.log"
	# delete the old logfile
	if os.path.isfile(logfilename):
		os.remove(logfilename)
		print("[info] logfile erased! '{}'".format(logfilename))
	logging.basicConfig(filename=logfilename, level=logging.DEBUG)
	logger = logging.getLogger(__name__)
	# avoid rendering when there are many simulation episodes
	if (nbr_episodes > 1):
		#rmode = 'console'
		rmode = None
	else: rmode = 'human'
	# register the env. with its parameters (kwargs)
	gym.register(
		 id="VacuumCleanerWorld-v0",
		 entry_point="vacuum.world:VacuumCleanerWorldEnv",
		 #episode_max_steps=300,		# the duration the vaccum cleaner robot is powered on
		 # env args default values
		 kwargs={'grid':None, 'dirt_comeback':dirt_flag, 'dirt_proba':0.09, 'murphy_proba':0.11, 
		 'location_sensor':location_sensor, 'episode_max_steps':max_steps, 'render_mode':rmode}
	)
	# converted the selected vacuum map to 2D numpy array
	wmap = np.array(world_map)
	# uncomment if you prefer 'console' render mode when there are many episodes
	#env = gym.make('VacuumCleanerWorld-v0', grid=wmap, render_mode='console')
	# create the gym env.
	env = gym.make('VacuumCleanerWorld-v0', grid=wmap)
	# seed the env, 
	# uncomment the next line for reproducible results
	# or for a single episode comparison
	env.reset(seed=SEED)
	env.unwrapped.set_map_name(world_id)			# for GUI rendering
	logger.info(str(env.spec.kwargs))
	# add a wrapper to truncate simulation after episode_max_steps
	# TimeLimit unneeded, since I handle simulation termination condition in 
	# the env code using episode_max_steps
	#env = TimeLimit(env)		# episode_max_steps specified in register
	print("[info] Observation space: ", env.observation_space)
	print("[info] Action space: ", env.action_space)
	reward_dict = env.unwrapped.get_rewards()
	action_dict = env.unwrapped.get_actions()
	print("[info] Actions dictionary: ", action_dict)
	print("[info] Rewards dictionary: ", reward_dict)
	print('[info] Location sensor: ', location_sensor)
	eco_flag = False # economic operation mode				
	# a greedy policy stops re-exploring rooms when the agent knows 
	# that it cleaned all rooms and dirt won't comeback
	if not dirt_flag and policy_id in [1,2]:	
		# when greedy or greedy random policy selected
		answer = input("[prompt] does the agent knows that dirt won't comeback? [y/n]")
		if answer == 'y':	
			eco_flag = True      # to be rational
	# create and reset the selected cleaning policy
	policy = Tools.make_policy(policy_id, world_id, env, eco_mode=eco_flag)
	policy.reset(seed=SEED)

	env.unwrapped.set_agent_name(policy_id)			# for GUI randering
	print("[info] Policy: {}".format(policy_id))
	print('[info] Map ID: {}'.format(world_id))
	print('[info] dirt comeback: ', dirt_flag)
	print('[info] simulating {} episodes...'.format(nbr_episodes))
	logger.info("[info] map:'{}'' policy:'{}' location_sensor:'{}' eps:{}\
		 max_steps:{}".format(world_id, policy_id, location_sensor,\
		 	nbr_episodes, max_steps))
	key = input("[prompt] press 'Enter' to start simulation!")
	if key != "": exit()
	# initialize the agent performance metrics per episode
	rewards = np.zeros(nbr_episodes)
	cleanings = np.zeros(nbr_episodes)
	travels = np.zeros(nbr_episodes)
	# simulation main loop
	if  isinstance(policy, QLearnPolicy):

		EPISODES = 1000
		if policy.load_qtable():
			print(f"Q-table chargée pour la carte {world_id}")
		else:
			print(f"Entraînement pour la carte {world_id} sur {EPISODES} épisodes...")
			policy.train_q_learning(env, episodes=EPISODES)
			print(f"Q-table sauvegardée pour la carte {world_id}")

	for eps in range(nbr_episodes):
		print("[info] episode {}, world configuration: ".format(eps+1))
		# reset the agent  policy and environment
		policy.reset()
		state = env.reset()[0]
		print("Initial state: (agent coordinates, dirt)=({},{}), world map: \n {}".\
			format(state['agent'], state['dirt'], wmap))
		print("step \t action  reward  state(ag_loc,dirt) \t info(dirty, act_done)")		
		done = False
		# espisode sim loop

		while not done:
			#if (policy_id == 3):
			#	EPISODES=1000
			#	if policy.load_qtable():
			#		print(f"Q-table chargée pour la carte {world_id}")
			#		action = policy.select_action(state)
			#	else:
			#		print(f"Entraînement pour la carte {world_id}...")
			#		policy.train_q_learning(episodes=1000)
			#		print(f"Q-table sauvegardée pour la carte {world_id}")

			#else:
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
			print("[info] mission accomplished! (all the rooms are clean)")
		else:
			print("[info] mission failed! (dirty rooms left)")
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
	# save results for comparison or reports
	# only when the simulation has a fair number of episodes
	if nbr_episodes<50:
		exit()
	if input("[prompt] save results? [y/n]") == 'y':
		results = {'reward':rewards, 'cleaned':cleanings, 'travel':travels}
		Tools.save_results(world_id, policy_id, results)
	print('[info] to plot results, type: ')
	print('[info] python vacuum.py -v [world_id]')

def print_help():
	"""
	display script's help and examples
	"""
	world_ids = Map.get_world_ids()		# from 'maps.py'
	policies = Tools.get_policies()
	py_help = f"""
	Usage:  
		python main.py [-h] map policy [-e episodes episode_maxsteps] [-ls] [-cl]
			Runs the vacuum cleaner gym environment.
			map: 	vacuum map number (0 ... {len(world_ids)-1})
			policy:	cleaning strategy number (0 ... {len(policies)-1})
			-e	sets 'episodes' and 'episode_maxsteps'
			episodes: number of simulation repetition (default: 1)
			episode_maxsteps: number of times the MDP is ran? (default: 100)
			-ls sets the location sensor as faulty/abscent (default: working), 
			-cl clean flag. When set the rooms remain clean after dusting 
				(default: dust comebacks)
			-h displays this Help.
		
		python main.py -v [map_nbr]
			Displays, on terminal console, the pre-defined vacuum world maps 
			or some given map.
			map_nbr:	the id number of the map to display

		python -m vacuum.tools -v [world_id]
			shows comparative results plots 
			
	Examples:
		'python main.py 1 0' 
			Simulates one episode with 100 steps of 'greedy' policy on 
			'vacuum-2rooms' map, dirt reappear and the agent can sense
			its location.
		'python main.py 0 0 -e 5 1000' 
			Simulates 'random' clean policy on 'vacuum-2rooms' map over 5 
			episodes each with 1000 steps, the agent location sensor works 
			and dust may reappear.
		'python main.py 1 2 -ls'	
			Simulates 'greedy' policy on 'vacuum-3rooms-v0' map during a single  
			episode with 100 steps, the robot can't be informed where it is, 
			dirt comebacks.
	"""
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

"""
parse command args
:param args: command line arguments
:returnn: map id, policy id, number of episodes, max steps per episode, 
		  dirt repearance flag, location sensor presence flag.
 
"""
def parse_commandline_args(args):
	nbr_args = len(args)
	if nbr_args==1 or nbr_args==2 and args[1] == '-h':
		print_help()
		exit()
	world_ids = Map.get_world_ids()		# from 'maps.py'
	# display a pre-defined world map designated or all maps
	if args[1] == '-v':
		if nbr_args == 3:
			mapid = world_ids[int(args[2])]
			Map.display_map(mapid)
		else:
			Map.display_maps()
		exit()
	policies = Tools.get_policies()
	location_sensor = True              # the location sensor works (default)
	dirt_flag = True                    # dirt comeback after cleaning (default)
	policy_n = None					# policy identifier number
	try:
		# eg. python testme.py 1 1 -e 10 200 -ls -cl
		if (nbr_args >= 3):
			# make sure map and policy selection is correct
			if int(args[1]) not in range(len(world_ids)): 
				raise ValueError("map #{} doesn't exist!".format(args[1]))
			if int(args[2]) not in range(len(policies)):
				raise ValueError("policy #{} doesn't exist!".format(args[2]))
			world_id = world_ids[int(args[1])]
			policy_n = int(args[2])
			for key in policies:
				if policies[key] == policy_n:
					policy_id = key
					break
			nbr_episodes = EPISODES    # default value
			max_steps = STEPS          # default value
		else:
			print("[error] select a map and a policy!")
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
	return world_id, policy_id, nbr_episodes, max_steps, dirt_flag, location_sensor 

# main program call		
main()