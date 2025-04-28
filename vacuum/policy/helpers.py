"""
'vacuum.policy.helpers.py'
-----------
Vacuum cleaner world, 2024.
Helper functions as for listing available policies and 
creating a given cleaning policy
Hakim Mitiche
March 2024
"""
#from vacuum.policy.base import RandomPolicy
from vacuum.policy.base import RandomPolicy				# relative module naming
from vacuum.policy.qlearning import QLearnPolicy
from vacuum.policy.greedy import GreedyPolicy
from vacuum.policy.greedyrandom import GreedyRandomPolicy

def make_policy(policy_id, world_id, env, eco_mode):
	"""
	Instantiates a cleaning policy
	Parameters:
		policy_id (Int): policy identifier, see: 'main.py'
		world_id (String): map identifier
		eco_mode (Boolean): flag for economic mode. 
							when set, the agent stops checking rooms 
							if told that dirt won't comeback.
	Return: a cleaning policy
	"""
	pdict = get_policies()
	match (pdict[policy_id]):
		case 0:	p = RandomPolicy(env)
		case 1: p = GreedyRandomPolicy(world_id, env, eco=eco_mode)
		case 2:	p = GreedyPolicy(world_id, env, eco=eco_mode)
		case 3:	p = QLearnPolicy(world_id, env)
		case _:
			raise ValueError(f"Incorrect policy identifier number {policy_id}!")
			#self.logger.critical("Incorrect policy id!")
			return
	return p


def get_policies():
	"""
	Returns a dictionary with cleaning policies names as keys and 
	policies identifying number as values. These are the policies 
	you defined in vacuum.policy based CleanPolicy in 'base.py'
	"""
	return{
		"random": 0,			# pure random, reflex-based agent
		"greedy-random": 1,		# greedy with some randomness, reflex-based agent
		"greedy": 2,			# greedy, usually a relex-based agent with model	
		"q-learning": 3,		# Q-learning, a learning-based agent
	}