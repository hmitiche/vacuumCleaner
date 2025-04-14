#from .base import CleanPolicy

#class QLearnPolicy(CleanPolicy):

	#def __init__(self, wolrd_id, env):
	#	raise NotImplementedError

	#def select_action(self, state):
		#raise NotImplementedError

from world import VacuumCleanerWorldEnv
from policy.base import CleanPolicy
class QLearnPolicy(CleanPolicy):

	def __init__(self, wolrd_id, env):
		super().__init__("q-learning", world_id, env)
		self._location=Map.location_list(wolrd_id)
		assert self._locations is not None
		self._env=env

	def select_action(self, state):
		"""Choisit une action selon la politique ε-greedy."""
		if np.random.uniform(0, 1) < self.epsilon:
		# Exploration: choisir une action aléatoire
			return self.env.action_space.sample()
		else:
			# Exploitation: choisir l'action avec la valeur Q maximale
			return np.argmax(self.q_table[state])

	# Q-Learning Training with Epsilon-Greedy and Linear Epsilon Decay
	def train_q_learning(self,env, episodes, learning_rate_a=0.1, discount_factor_g=0.95, epsilon=1.0, epsilon_min=0.01,
						 epsilon_decay_rate=0.001):
		self.q_table = np.zeros((env.map_size, env.map_size,
								 2, env.action_space.n))
		self._episode_reward = 0



		for episode in range(episodes):
			state = env.reset()[0]

			while(not terminated and not truncated):
				action = self.select_action(state)
				(next_obs, reward, terminated, truncated,
				 info) = self.env.step(action)
				next_state = self.get_state_index(next_obs)
				q[state, action] = q[state, action] + learning_rate_a * (
						reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
				)

				self._episode_reward += reward
				state = next_state
			epsilon = max(epsilon_min, epsilon - epsilon_decay_rate)




