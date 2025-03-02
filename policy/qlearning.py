from policy.base import CleanPolicy

class QLearnerPolicy(CleanPolicy):

	def __init__(self, wolrd_id, env):
		raise NotImplementedError

	def select_action(self, state):
		raise NotImplementedError	