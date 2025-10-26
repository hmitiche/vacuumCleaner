# file: termination_wrapper.py
"""
Define a wrapper of VacuumCleanerWorldEnv (vacuum.world) to terminate an episode 
after all rooms are cleaned under the assumption that dirt won't comeback, 
that is, dirt_comeback should be set to False
"""
import gym
from gym import Wrapper

class TerminationWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        assert not env.unwrapped.dirt_comeback,\
        f"""
        [AssertionViolation] you can't terminate an episode after cleaning all rooms for 
        dirt may comeback! 
        """

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if self.env.unwrapped.count_rooms(clean=True) == self.env.unwrapped.nbr_rooms:
        	# stop when all rooms have been visited and cleaned
            done = True
            info['terminated_by_wrapper'] = True

        return obs, reward, done, truncated, info