import copy

import numpy as np
import gym

from src.gym.gym.utils import seeding
from src.gym.gym.vector.utils import spaces


class Knapsack(gym.Env):
    """
    An environment for knapsack problem; we have an empty knapsack with a limited mass capacity, and we want to fill it
    in with different items chosen from certain categories, where ach category has a specific fixed mass and value.
    For example, we can fill a knapsack with capacity of 1.5 kg with 50 x letter + 1 x small box, where letter and box
    are representing two categories with fixed
     - mass of 10 gr and 1 kg,
     - monetary value of 28 cents and 2 euros.
    This combination results in 16 euros ( = 50 x 28 cents + 2 euros).

    The goal is to pick items such that total sum of values of packed items is maximized while their total mass remains
    below the knapsack's mass capacity.

    Actions:
    Type: Discrete (N), where N is the number of items


    Description                                     Shape              Range
    ------------                                    -------            -------
    The item id to be chosen                        (1,)               integers [0, N)

    Note: In case the chosen item is not available (i.e. the availability is zero), that action is effectively ignored,
    i.e.
    - the state is not changed,
    - the reward is zero, and
    - the episode continues to the next step.

    Observation:
    Type: Dict

    Observation                                                    Key                  Shape            Range
    -----------                                                    -----                -------         -------
    currently packed mass                                         'used capacity'       (1,)             [0, capacity]
    number of available samples for each item to choose from      'availability'        (N,)             taken from conf

    Reward:
    The specific profit for the chosen item. In case none of that item is available, the reward would be zero.

    Reset:
    A random state is created with:
    - an empty knapsack, i.e. zero value for currently packed mass (used_capacity <- 0),
    - a randomly generated 'availability' list of integers, based on min and max given as configs

    Episode Termination:
    When the agent attempts going beyond the available mass capacity. In that case:
    - the state is not affected with this last action,
    - the episode is ended,
    - reward zero is returned
    Or
    When all the available items are gone (there is nothing left to pick up).
    """

    def __init__(self, env_config):
        """
        Instantiates an environment with based on the given config.
        :param env_config: a pyhocon object which has the following attributes:
        - capacity: the total mass capacity of the knapsack; a float
        - specific_mass: a list of float indicating the mass of each category.
        - specific_value: the monetary value for each category
        - max_ / min_availability: two lists of integers, indicating the typical min and max number of available items
        for each category at the beginning of the episode.
        - seed: the random seed; either an integer or None. In case None is given, the seed is not fixed.
        """
        super(Knapsack, self).__init__()
        # 0 - offloading the configs
        self.conf = env_config['conf']
        # the total capacity
        capacity = self.conf.capacity

        # typical max and min number of available items for each category
        self.min_availability = np.array(self.conf.min_availability, dtype=np.float32)
        self.max_availability = np.array(self.conf.max_availability, dtype=np.float32)
        self.seed_value = None if self.conf.seed_value == 'None' else self.conf.seed_value
        # 1 - Observation space
        self.observation_space = spaces.Dict(
            {'availability': spaces.Box(low=self.min_availability, high=self.max_availability, dtype=np.float32),
             'used capacity': spaces.Box(low=np.array([0]), high=np.array([capacity]), shape=(1,), dtype=np.float32)
             })

        # 2 - Action space
        self.action_space = spaces.Discrete(self.conf.nr_categories)

        # 3 - Seeding
        self.state = None
        self.done = None
        self.np_random = None
        self.seed(self.seed_value)
        self.reset()

    def seed(self, seed_value=None):
        """
        Sets the seed of the random generator
        :param seed_value: the value for the seed, if None is given the
        :return:
        """
        if seed_value is not None:
            self.np_random, seed = seeding.np_random(seed_value)
        else:
            self.np_random, seed = seeding.np_random()

        return [seed]

    def reset(self):
        """
        Resets the state to a random state and the variable done to False.
        :return: the state
        """
        availability = [self.np_random.randint(low=self.min_availability[i],
                                               high=self.max_availability[i] + 1) for i in range(self.nr_categories)]
        self.state = {'availability': np.array(availability, dtype=np.float32),
                      'used capacity': np.array([0], dtype=np.float32)}
        self.done = False
        return self.state

    def step(self, action):
        """
        Takes one step of packing with the given action. Here the action points to which category to pack from next.
        In case the action is not admissible (there are zero items in the chosen category available), nothing happens.
        In case we pack beyond the capacity or there is nothing left to pack, the process is terminated.
        :param action: the taken action; an integer indicating from which category agent picks an item.
        :return: state, reward, done, info
        """
        done = False
        backup = copy.deepcopy(self.state)
        # Apply the action
        self.state['availability'][action] -= 1.0
        self.state['used capacity'][0] += self.masses[action]
        reward = self.values[action]
        # if the action was not possible
        if self.state['availability'][action] < 0.:
            self.state = backup
            reward = 0
        # Terminate if:
        # - all the available items are gone (in this case take the reward calculated before).
        # - we went beyond the capacity. in this case revert the action and reward zero.
        if all(self.state['availability'] == 0):
            done = True
        if self.state['used capacity'][0] > self.capacity:
            self.state = backup
            done = True
            reward = 0

        return self.state, reward, done, {}

    def render(self, mode):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
    @property
    def nr_categories(self):
        """
        :return: the number of categories.
        """
        return self.conf.nr_categories

    @property
    def capacity(self):
        """
        :return: the mass capacity of the knapsack.
        """
        return self.conf.capacity

    @property
    def masses(self):
        """
        :return: the masses of the categories
        """
        return self.conf.specific_masses

    @property
    def values(self):
        """
        :return: the monetary values of the categories
        """
        return self.conf.specific_values

    @property
    def available_capacity(self):
        """
        :return: the empty space.
        """
        return self.capacity - self.state['used capacity'][0]
