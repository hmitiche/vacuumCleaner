"""
'tools.py'
-----------
Vacuum cleaner world, 2024.
A class with some util functions (static), which I couldn't find where to fit 
elsewher, such as for saving a simulation output, plotting the results or 
launching a cleaning policy.
Hakim Mitiche
March 2024
"""
#from vacuum.policy.base import RandomPolicy
from .policy.base import RandomPolicy				# relative module naming
from vacuum.policy.qlearning import QLearnPolicy
from vacuum.policy.greedy import GreedyPolicy
from vacuum.policy.greedyrandom import GreedyRandomPolicy
from vacuum.maps import Map
import matplotlib.pyplot as plt
import os, os.path
import pickle
import sys
import logging

# @improve_me: move these constants to a separate file: 'constants.py'
DATA_PATH = 'data/'
PLOT_PATH = 'plots/'
LOG_PATH = 'log/'


class Tools:
	"""
	Some util functions such as for loading a cleaning policy, 
	saving simulation results and ploting.
	"""
	
	logger = None
	@staticmethod
	def make_policy(policy_id, world_id, env, eco_mode):
		"""
		Instantiates a cleaning policy
		Parameters:
			policy_id (Int): policy identifier, see: 'main.py'
			world_id (String): map identifier
			eco_mode (Boolean): flag for economic mode. 
								when set, the agent stops checking rooms 
								if told that dirt won't comeback.
		"""
		pdict = Tools.get_policies()
		match (pdict[policy_id]):
			case 0:	p = RandomPolicy(env)
			case 1: p = GreedyRandomPolicy(world_id, env, eco=eco_mode)
			case 2:	p = GreedyPolicy(world_id, env, eco=eco_mode)
			case 3:	p = QLearnPolicy(world_id, env)
			case _:
				raise ValueError(f"Incorrect policy number {policy_id}!")
				#self.logger.critical("Incorrect policy id!")
				return
		return p

	"""
	Returns commun and user-defined vacuum cleaning policies with their codes (numbers).
	The user can add other policies here
	and define their classes by extending 'policy.CleanPolicy'.
	"""
	@staticmethod
	def get_policies():
		return{
			"random": 0,			# pure random, reflex-based agent
			"greedy-random": 1,		# greedy with some randomness, reflex-based agent
			"greedy": 2,			# greedy, usually a relex-based agent with model	
			"q-learning": 3,		# Q-learning, a learning-based agent
		}

	"""
	Converts a list to a string with some formating for console output. 
	:param: l 		some list of items
	:param: sep 	items string separator
	:param: freq 	number of items to display per line
	:param: margin  number of tabs that defines the left margin 
	"""	
	@staticmethod
	def str_list(l, freq=None, margin=1, sep=", "):
		count = 0
		strlist = ""
		for e in l:
			strlist = strlist + e + sep
			count += 1
			if freq is not None and count%freq == 0:
				strlist = strlist + "\n" + "\t"*margin
		# removes the last sep or line skip and margin
		if strlist[-len(sep):] == sep:
			return strlist[:-len(sep)]
		return strlist[:-(1+margin)]
		
	"""
	Save results to file: 'data/<world_id>.pkl'.
	The data are used later to plot and compare policies.
	File contains a list of results dictionaries, where:
	key: policy_id, 
	value: dictionary of results (rewards:[], cleanings:[], travels:[]).
	The user needs to make sure the policies ve been tested with 
	the same configuration: max_steps, nbr_episodes, probas, flags, ...
	:param: world_id 	map identifier
	:param: policy_id	cleaning agent name
	:param:	results 	agent simulation results dictionary	
	"""
	@staticmethod
	def save_results(world_id, policy_id, results):
		logger = Tools.init_logger(world_id)
		datafile = '{}results_{}.pkl'.format(DATA_PATH, world_id)
		results_dict = None
		# load previous results, if any
		if os.path.exists(datafile):
			with open(datafile, 'rb') as file:
				results_dict = pickle.load(file)
				file.close()
		# open/creates results binary file with write access
		file = open(datafile, 'wb')
		# am I saving the first policy result?
		if results_dict is None:
			results_dict = dict({})		
		results_dict[policy_id] = results  # update old results, if any
		pickle.dump(results_dict, file)
		print("[info] {} results saved to {}".format(policy_id, datafile))
		file.close()
		 
	"""
	Plot vaccum cleaning policies performance on the given world map.
	Plot each metric separatedly.
	:param: world_id 	vacuum world map name
	:param: scattered 	if True, use dots, else use lines.
	:param: animapted 	animate the plot.
	"""

	@staticmethod
	def plot_results(world_id, scattered=False, animated=False):
		logger = Tools.init_logger(world_id)
		datafile = '{}results_{}.pkl'.format(DATA_PATH, world_id)
		if os.path.exists(datafile):
			with open(datafile, 'rb') as file:
				data = pickle.load(file)
		else:
			raise FileNotFoundError(f"Couldn't find file {datafile}")
			#logger.error("{} not found! nothing plotted".format(datafile))
			exit()
		if data == None or data == {}:
			logger.error(f"No results in file: {datafile}")
			raise ValueError()
			return
		policies = list(data.keys())
		#logger.info(policies)
		print(f"policies: {policies}")
		# I assume policies results ve the same metrics
		# number of graphs (axes) is the number of policies
		metrics = list(data[policies[0]].keys())	# results metrics names
		print(f"metrics: {metrics}")
		#logger.info(metrics)
		nbr_plots = len(metrics)
		#logger.info(nbr_plots)
		pause = 0.2							# animation pause in ms
		eps = len(data[policies[0]][metrics[0]])	# nbr episodes
		print(f"episodes: {eps}")
		x = [i for i in range(1, eps+1)]
		colors = ["blue", "orange", "green", "black"]
		y = [[[d for d in data[p][m]] for p in policies] for m in metrics]
		for i in range(nbr_plots):
			plt.clf()
			m = metrics[i]
			plt.title(f"World Map '{world_id}': {m}")
			plt.xlabel('episode')
			plt.ylabel(metrics[i])
			plt.xlim(0,eps+2)
			# progressively fill subplots, one by one
			if not animated:
				#plt.ylim(y[i][])
				for j in range(len(policies)):
					if not scattered:
						plt.plot(x, y[i][j], label=policies[j])
					else:
						plt.scatter(x, y[i][j], label=policies[j])
			else:	
				for e in range(eps):
					for j in range(len(policies)):
						p = policies[j]
						if not scattered:
							plt.plot(x[:e],y[i][j][:e],label=p, color=colors[j])
						else:
							plt.scatter(x[:e],y[i][j][:e],label=p, color=colors[j])
					plt.pause(pause)
					if e == 1:
						plt.legend(policies, loc='best')
			plt.legend(policies, loc='best')
			figure_name = '{}Figure_{}_{}.png'.format(PLOT_PATH, world_id, m)
			print("[info] plot save to '{}'".format(figure_name))
			plt.savefig(figure_name)
			plt.show()
			inp = input("[prmpt] press 'Enter' to show next plot ...")
			if inp != "": exit()
		#input("press any key to close the plot")
		# using show won't let the plot be saved to a file
		"""
		figure_name = '{}plot_{}.png'.format(LOG_PATH, world_id)
		if (input("save figure? [Y/n]") == 'y'):
			plt.savefig(figure_name)
			print("plot save to '{}'".format(figure_name))
			print("type: open ", figure_name)
		"""

	# plot results using subplots of all the metrics
	#@fix_me: I think it's not working, it's not useful sofar.	
	@staticmethod
	def subplot_results(world_id):
		logger = Tools.init_logger(world_id)
		datafile = '{}results_{}.pkl'.format(DATA_PATH, world_id)
		if os.path.exists(datafile):
			with open(datafile, 'rb') as file:
				data = pickle.load(file)
		else:
			raise FileNotFoundError(f"Couldn't find file {datafile}")
			#logger.error("{} not found! nothing plotted".format(datafile))
			exit()
		if data == None or data == {}:
			logger.error(f"No results in file: {datafile}")
			raise ValueError()
			return
		policies = list(data.keys())
		logger.info(policies)
		# I assume policies results ve the same metrics
		# number of graphs (axes) is the number of policies
		metrics = list(data[policies[0]].keys())	# results metrics names
		logger.info(metrics)
		nbr_plots = len(metrics)
		logger.info(nbr_plots)
		#plt.title(f"Vacuum world {world_id}")
		pause = 0.5							# animation pause in ms
		fig, ax = plt.subplots(nbr_plots, 1, sharex=True)
		#logger.info(y)
		eps = len(data[policies[0]][metrics[0]])	# nbr episodes
		x = [i for i in range(eps)]
		y = [[0 for i in range(eps)] for j in range(len(policies))]
		for i in range(nbr_plots):
			m = metrics[i]
			#ax[i].set_xlabel('episode')
			ax[i].set_ylabel(metrics[i])
			# progressively fill subplots, one by one
			#x = []
			#y = [[] for j in range(len(policies))]
			for e in range(eps):
				#x.append(e+1)
				lg = ['' for j in range(len(policies))]
				for j in range(len(policies)):
					#fig.clf()
					p = policies[j]
					#y[j].append(data[p][m][e])
					y[j][e] = data[p][m][e]
					lg[j],=ax[i].plot(x,y[j],label=p)
				plt.pause(pause)
			#ax[i].legend(loc='best')
			#input("press any key to show next plot")
		# legend on the last plot
		#ax[nbr_plots-1].legend(policies, loc='best')
		# title on the first plot
		ax[0].set_title(f"Vacuum world {world_id}")
		ax[nbr_plots-1].set_xlabel('episode')
		fig.legend((tuple(lg)), tuple(policies))
		plt.show()
		input("press any key to close the plot")

	@classmethod
	def init_logger(cls, world_id):
		if cls.logger is not None:
			return cls.logger
		logfile = f"{LOG_PATH}{world_id}.log"
		# delete old simulation logfile
		#if os.path.isfile(logfile):
		#	os.remove(logfile)
		#	print("[info] logfile erased: {}".format(logfile))
		logging.basicConfig(filename=logfile, level=logging.DEBUG)
		cls.logger = logging.getLogger(__name__)
		return cls.logger

# command entry program
if __name__ == '__main__':
	help_msg = f"""
	Usage:	python tools.py -h 
				display this help.
				
			python tools.py -v world_id [-s] [-a]
				show a comparaison plot of vacuum cleaning policies for 
				world_id and/or save it to figure: 'Figure_[world_id]_[metric].png'
				world_id: {Tools.str_list(Map.get_world_ids(), freq=4, margin=3, sep=", ")}
				-s do scatter plots (default: lines)
			-a animate the plots (default: false)

	Examples:
		python tools.py -v vacuum-2rooms-v0
			plot results (rewards, cleaned, travels, ...) 
			of policies {Tools.str_list(list(Tools.get_policies().keys()))}
			on map 'vacuum-2rooms-v0'
		python tools.py -v vacuum-2rooms-v0 -s -a
			genearte an animated plot using scattered dots	
	"""
	args = sys.argv
	narg = len(args)
	if narg==1 or args[1] == '-h':
		print(help_msg)
		exit()
	if narg>=3 and args[1]=='-v':
		world_id = args[2]
		# in case the user mistaped
		assert world_id in Map.get_world_ids(), "Please enter a valid map id!"
		if narg >= 4 and args[3] == '-s': scatter = True
		else: scatter= False
		if narg ==4 and args[3] == '-a' or narg==5 and args[4] == '-a': 
			anim = True
		else: anim= False
	else:
		print("[error] command incorrect!")
		print(help_msg)
		exit()		
	Tools.plot_results(world_id, scatter, anim)