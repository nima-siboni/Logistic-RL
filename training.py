from pyhocon import ConfigFactory
from environment.knapsack import Knapsack
from utils.utils import conf_value_checking

# 0 - Reading and sanity check of the configs
conf = ConfigFactory.parse_file('meta_conf.conf')
conf_value_checking(conf)
# 1 - Creating the environment
env = Knapsack(env_config={'conf': conf.Env})

# 2 - Training goes here!
