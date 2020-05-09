import yaml

from algorithms.dqn.dqn import DQN
from cpprb import PrioritizedReplayBuffer
from replay_buffers.util import DictWrapper, get_dtype_dict
from algorithms.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *
import tensorflow as tf
import os
from common.tf_util import config_gpu


def make_env(frame_skip, frame_stack, stack_key='pov', **kwargs):
    env = RozumEnv(**kwargs)
    if frame_skip:
        env = FrameSkip(env, frame_skip)
    if frame_stack:
        env = FrameStack(env, frame_stack, stack_key=stack_key)
    env = AccuracyLogWrapper(env, 10)
    discrete_dict = dict()
    robot_dof = env.action_space.shape[0]
    for i in range(robot_dof):
        discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
        discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
    env = DiscreteWrapper(env, discrete_dict)
    return env


def dqn_run(config_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.config.optimizer.set_jit(True)
    config_gpu()
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    env_config = config['env']
    env = make_env(**env_config)
    env_dict, dtype_dict = get_dtype_dict(env)

    buffer_config = config['buffer']
    replay_buffer = PrioritizedReplayBuffer(env_dict=env_dict, **buffer_config)
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_keys = env.observation_space.spaces.keys()
        replay_buffer = DictWrapper(replay_buffer, state_prefix=('', 'next_', 'n_'),
                                    state_keys=state_keys)
    make_model = get_network_builder(config['nn_model'])
    agent_config = config['agent']
    agent = DQN(replay_buffer, make_model, env.observation_space, env.action_space, dtype_dict,
                **agent_config)

    train_config = config['train']
    summary_writer = tf.summary.create_file_writer(train_config.pop('log_dir'))
    with summary_writer.as_default():
        agent.train(env, **train_config)
    env.close()
