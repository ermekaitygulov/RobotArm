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


def make_env(frame_skip, frame_stack, stack_key='pov', **kwargs):
    env = RozumEnv(**kwargs)
    if frame_skip > 1:
        env = FrameSkip(env, frame_skip)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack, stack_key=stack_key)
    env = RozumLogWrapper(env, 10)
    mu = np.zeros_like(env.action_space.low)
    sigma = np.ones_like(env.action_space.low) * 0.01
    sigma[-1] = 0.1
    theta = np.ones_like(env.action_space.low) * 0.15
    theta[-1] = 0.1
    env = CorrelatedExploration(env, mu, sigma, theta)
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
    nn_config = config['neural_network']
    make_critic = get_network_builder(nn_config['critic'])
    make_actor = get_network_builder(nn_config['actor'])

    agent = DDPG(make_critic, make_actor, env.observation_space, env.action_space,
                 replay_buff=replay_buffer, dtype_dict=dtype_dict, **config['agent'])

    train_config = config['train']
    summary_writer = tf.summary.create_file_writer(train_config.pop('log_dir'))
    with summary_writer.as_default():
        agent.train(env, **train_config)
    env.close()
