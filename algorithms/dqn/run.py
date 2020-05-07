from algorithms.dqn.dqn import DQN
from cpprb import PrioritizedReplayBuffer
from replay_buffers.util import DictWrapper, get_dtype_dict
from algorithms.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *
import tensorflow as tf
import os
from common.tf_util import config_gpu


def make_env(env):
    env = FrameSkip(env)
    env = FrameStack(env, 2, stack_key='pov')
    env = AccuracyLogWrapper(env, 10)
    discrete_dict = dict()
    robot_dof = env.action_space.shape[0]
    for i in range(robot_dof):
        discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
        discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
    env = DiscreteWrapper(env, discrete_dict)
    return env


def dqn_run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.config.optimizer.set_jit(True)
    config_gpu()

    env = RozumEnv()
    env = make_env(env)
    env_dict, dtype_dict = get_dtype_dict(env)
    replay_buffer = PrioritizedReplayBuffer(size=100000, env_dict=env_dict)
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_keys = env.observation_space.spaces.keys()
        replay_buffer = DictWrapper(replay_buffer, state_prefix=('', 'next_', 'n_'),
                                    state_keys=state_keys)
    make_model = get_network_builder("DuelingDQN_pov_angle")
    agent = DQN(replay_buffer, make_model, env.observation_space, env.action_space, dtype_dict,
                replay_start_size=100, train_quantity=100, train_freq=100, log_freq=20)
    summary_writer = tf.summary.create_file_writer('train/')
    with summary_writer.as_default():
        agent.train(env, 1000)
    env.close()


if __name__ == '__main__':
    dqn_run()
