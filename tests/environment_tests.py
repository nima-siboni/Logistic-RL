# testing the environment
# some dirty tests
import copy
import pytest
import numpy as np
from pyhocon.config_parser import ConfigFactory

from environment.knapsack import Knapsack
from utils.utils import conf_value_checking

# tolerance for numerical rounding errors
EPSILON = 1e-6


@pytest.fixture
def config():
    # Reading and sanity check of the configs
    conf = ConfigFactory.parse_file('meta_conf.conf')
    conf_value_checking(conf)
    return conf.Env


def test_environment_init(config):
    # testing the environment initialization
    env = Knapsack(env_config={"conf": config})
    # testing the conf
    assert env.conf == config
    # testing the state
    assert list(env.state.keys()) == ['availability', 'used capacity']
    assert env.state['used capacity'] == 0
    # testing availability smaller than max_availability
    assert (env.state['availability'] <= env.max_availability).all()
    # testing availability larger than min_availability
    assert (env.state['availability'] >= env.min_availability).all()
    # testing done
    assert not env.done


def test_step_function(config):
    # testing the step function
    env = Knapsack(env_config={"conf": config})
    # 1. Testing the step function with a valid action
    # check the change of the state in case we take a valid action
    # we expect that the used capacity is increased by the amount of the specific mass of the chosen item
    # and the availability is decreased by the amount of the specific mass of the chosen item
    # and the reward is the specific value of the chosen item
    assert (env.state['availability'] == np.array([62., 78., 52., 77., 48., 17., 4., 3., 3., 1.])).all()
    assert env.capacity == 70
    action = 0
    nr_item_for_the_action = int(env.state['availability'][action])
    for i in range(nr_item_for_the_action):
        _, reward, _, _ = env.step(action=action)
        assert (env.state['availability'] ==
                np.array([nr_item_for_the_action - i - 1., 78., 52., 77., 48., 17., 4., 3., 3., 1.])).all()
        assert reward == config.specific_values[action]
        tmp = env.state['used capacity'] - np.array([(i + 1) * config.specific_masses[action]], dtype=np.float32)
        assert np.abs(tmp) < EPSILON
    # check if the state is done
    assert not env.done

    # 2. Testing the behavior of the environment in case we take an invalid action
    # deep copy the state for reference
    state_copy = copy.deepcopy(env.state)
    # taking the invalid action for a couple of times (10 here).
    for i in range(10):
        _, reward, done, _ = env.step(action=action)
        assert reward == 0
        assert not done
        assert (env.state['availability'] == state_copy['availability']).all()
        assert (env.state['used capacity'] == state_copy['used capacity']).all()
    # check if the state is done
    assert not env.done


def test_termination(config):
    # testing the termination
    env = Knapsack(env_config={"conf": config})
    # 1. Testing the termination
    # scenario 1: first taking the last item (item = 9) and then the 2 times item 8.
    # the state / action pairs are
    # avail: [62, 78, 52, 77, 48, 17, 4, 3, 3, 1], used: 0 with action = 9 to
    # avail: [62, 78, 52, 77, 48, 17, 4, 3, 3, 0], used: 31.5 with action = 8
    # avail: [62, 78, 52, 77, 48, 17, 4, 3, 2, 0], used: 51.5 with action = 8
    # avail: [62, 78, 52, 77, 48, 17, 4, 3, 2, 0], used: 51.5  and here we expect termination because the capacity is
    # exceeded if we take the action 8 (going from 51.5 to 70.5 which is bigger than 70)
    action = 9
    _, reward, done, _ = env.step(action=action)
    assert not done
    assert not env.done
    assert reward == env.conf.specific_values[action]
    assert (env.state['availability'] == np.array([62., 78., 52., 77., 48., 17., 4., 3., 3., 0.])).all()
    assert env.state['used capacity'] == 31.5

    action = 8
    _, reward, done, _ = env.step(action=action)
    assert not done
    assert not env.done
    assert reward == env.conf.specific_values[action]
    assert (env.state['availability'] == np.array([62., 78., 52., 77., 48., 17., 4., 3., 2., 0.])).all()
    assert env.state['used capacity'] == 51.5

    _, reward, done, _ = env.step(action=action)
    assert done
    assert env.done
    assert (env.state['availability'] == np.array([62., 78., 52., 77., 48., 17., 4., 3., 2., 0.])).all()
    assert env.state['used capacity'] == 51.5
    assert reward == 0
