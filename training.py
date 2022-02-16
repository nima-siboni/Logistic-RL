import ray
from pyhocon import ConfigFactory
import ray.rllib.agents.dqn as dqn
from environment.knapsack import Knapsack
from utils.utils import conf_value_checking

# 0 - Reading and sanity check of the configs
conf = ConfigFactory.parse_file('meta_conf.conf')
conf_value_checking(conf)
# 1 - Creating the environment

# 2 - Creating the agent
ray.init(ignore_reinit_error=True)
config = dqn.DEFAULT_CONFIG.copy()
config['num_workers'] = conf.training.num_workers
config['env_config'] = {'conf': conf.Env}
agents = dqn.DQNTrainer(config=config, env=Knapsack)
# 3. Training
for i in range(conf.training.nr_episodes):
    training_res = agents.train()
    # printing the min, max and average reward
    print('Episode: {}, Min: {:.2f}, Max: {:.2f}, Average: {:.2f}'.format(i, training_res['episode_reward_min'],
                                                                          training_res['episode_reward_max'],
                                                                          training_res['episode_reward_mean']))
    # saving the model
    if i % conf.training.save_model_freq == 0:
        agents.save(conf.training.save_model_path)
# 4. Saving the model
agents.save()
# 5. Shutting down the workers
ray.shutdown()
print('Done!')
