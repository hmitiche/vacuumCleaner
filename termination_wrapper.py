"""
Define a wrapper of VacuumCleanerWorldEnv (vacuum.world) to terminate an episode
after all rooms are cleaned under the assumption that dirt won't come back,
that is, dirt_comeback should be set to False
"""
from vacuum import world
import gymnasium
from gymnasium import Wrapper

class TerminationWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.total_rooms = None  # sera initialisé au reset

        assert not env.unwrapped.dirt_comeback, (
            "[AssertionViolation] You can't terminate an episode after cleaning all rooms "
            "because dirt may come back!"
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Maintenant que self.map est initialisé, on peut compter les pièces
        self.total_rooms = self.env.unwrapped.count_rooms()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if self.total_rooms is not None and self.env.unwrapped.count_rooms(clean=True) == self.total_rooms:

            reward +=5
            done = True
            info['terminated_by_wrapper'] = True
        else:
            reward-=2
        return obs, reward, done, truncated, info
