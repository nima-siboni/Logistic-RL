import numpy as np
import gym


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
        conf = env_config['conf']
        # the total capacity
        capacity = conf.capacity
        # typical max and min number of available items for each category
        min_availability = np.array(conf.min_availability)
        max_availability = np.array(conf.max_availability)
        # TODO: Offload the configs and implement the necessary functions.
