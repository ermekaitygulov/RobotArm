import yaml

from algorithms.ddpg.ddpg import DDPG
from cpprb import PrioritizedReplayBuffer as cppPER
from replay_buffers.stable_baselines import PrioritizedReplayBuffer
from replay_buffers.util import DictWrapper, get_dtype_dict
from algorithms.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *
import tensorflow as tf
import os
from common.tf_util import config_gpu


def make_env(mu=0., sigma=0.1, **kwargs):
    env = RozumEnv(**kwargs)
    env = RozumLogWrapper(env, 10)
    mu = np.ones_like(env.action_space.low) * mu
    sigma = np.ones_like(env.action_space.low) * sigma
    env = UncorrelatedExploration(env, mu, sigma)
    return env


def ddpg_run(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])

    tf.config.optimizer.set_jit(True)
    config_gpu()

    env = make_env(**config['env'])
    env_dict, dtype_dict = get_dtype_dict(env)

    if 'cpp' in config['buffer'].keys() and config['buffer'].pop('cpp'):
        dtype_dict['indexes'] = 'uint64'
        replay_buffer = cppPER(env_dict=env_dict, **config['buffer'])
    else:
        replay_buffer = PrioritizedReplayBuffer(env_dict=env_dict, **config['buffer'])
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_keys = env.observation_space.spaces.keys()
        replay_buffer = DictWrapper(replay_buffer, state_prefix=('', 'next_', 'n_'),
                                    state_keys=state_keys)
    network_kwargs = dict()
    for key, value in config['neural_network'].items():
        if isinstance(value, dict):
            network_kwargs[key] = get_network_builder(**value)
        else:
            network_kwargs[key] = get_network_builder(value)

    agent = DDPG(obs_space=env.observation_space, action_space=env.action_space,
                 replay_buff=replay_buffer, dtype_dict=dtype_dict, **config['agent'],
                 **network_kwargs)

    if 'pretrain_weights' in config:
        agent.load(**config['pretrain_weights'])
    if 'CriticViz' in config:
        env = CriticViz(env, agent.online_actor, agent.target_actor,
                        agent.online_critic, config['CriticViz'])
    train_config = config['train']
    summary_writer = tf.summary.create_file_writer(train_config.pop('log_dir'))
    with summary_writer.as_default():
        agent.train(env, **train_config)
    env.reset()
    env.close()
