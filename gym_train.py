import yaml
import algorithms
from nn_models.model import get_network_builder
from argparse import ArgumentParser
import tensorflow as tf

from common.tf_util import config_gpu
from common.wrappers import *
from replay_buffers.util import get_dtype_dict, DictWrapper
from cpprb import PrioritizedReplayBuffer as cppPER
from replay_buffers.stable_baselines import PrioritizedReplayBuffer
import os
import gym


def make_env(env_id, exploration_kwargs=None, frame_stack=4, discretize=True):
    exploration_kwargs = exploration_kwargs if exploration_kwargs else {}
    environment = gym.make(env_id)
    if frame_stack > 1:
        environment = stack_env(environment, frame_stack)
    if discretize:
        environment = make_discrete_env(environment, **exploration_kwargs)
    else:
        environment = make_continuous_env(environment, **exploration_kwargs)
    return environment


def make_continuous_env(environment, mu=0., sigma=0.1):
    mu = np.ones_like(environment.action_space.low) * mu
    sigma_vector = np.ones_like(environment.action_space.low)
    sigma_vector *= sigma
    environment = UncorrelatedExploration(environment, mu, sigma_vector)
    return environment


def make_discrete_env(environment, epsilon=0.1, final_epsilon=0.01, epsilon_decay=0.99):
    environment = EpsilonExploration(environment, epsilon, final_epsilon, epsilon_decay)
    return environment


def stack_env(environment, k):
    if isinstance(environment.observation_space, gym.spaces.Dict):
        for obs_key in environment.observation_space.spaces.keys():
            environment = FrameStack(environment, k, obs_key)
    else:
        environment = FrameStack(environment, k)
    return environment


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', action='store', help='Path to config with params for chosen alg',
                        type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
    tf.config.optimizer.set_jit(True)
    config_gpu()
    env = make_env(**config['env'])
    env_dict, dtype_dict = get_dtype_dict(env.observation_space, env.action_space)
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

    base = getattr(algorithms, config['base'])
    agent = base(obs_space=env.observation_space, action_space=env.action_space,
                 replay_buff=replay_buffer, dtype_dict=dtype_dict,
                 **config['agent'], **network_kwargs)
    if 'pretrain_weights' in config:
        agent.load(**config['pretrain_weights'])

    train_config = config['train']
    summary_writer = tf.summary.create_file_writer(config.pop('log_dir'))
    with summary_writer.as_default():
        agent.train(env, **train_config)
    env.reset()
    env.close()
