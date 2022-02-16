import copy
import gym
import numpy as np
from gym.vector.utils import spaces


class DictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """
        A wrapper around the environment to scale the state.
        :param env: the unscaled env
        """
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict), \
            "Should only be used to wrap Dict observations."
        self.unscaled_obs_space = copy.deepcopy(self.observation_space)

        scaled_obs_dict = {}
        # A loop going over all the keys of the observation dict
        for key in self.unscaled_obs_space:
            value = self.unscaled_obs_space[key]
            shape = value.shape
            low = np.zeros_like(value.low, dtype=np.float32)
            high = np.ones_like(value.high, dtype=np.float32)
            scaled_space = spaces.Box(low=low, high=high, dtype=np.float32)
            scaled_obs_dict[key] = scaled_space
        self.observation_space = spaces.Dict(scaled_obs_dict)

    def observation(self, obs):
        """
        scales all the observations between [0, 1]
        :param obs: the unscaled obs as a dict
        :return: scaled obs as a dict
        """
        scaled_obs = {}
        for key in obs:
            value = obs[key]
            high = self.unscaled_obs_space[key].high
            low = self.unscaled_obs_space[key].low
            scaled_values = (value - low) / (high - low)
            scaled_obs[key] = scaled_values
        return scaled_obs
