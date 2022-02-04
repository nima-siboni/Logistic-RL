from environment.knapsack import Knapsack
from pyhocon import ConfigFactory
conf = ConfigFactory.parse_file('meta_conf.conf')
# conf_value_checking(conf)
env = Knapsack(env_config={'conf': conf.Env})
print("good-bye")
