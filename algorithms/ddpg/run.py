from algorithms.ddpg.ddpg import DDPG
from cpprb import PrioritizedReplayBuffer
from replay_buffers.util import DictWrapper, get_dtype_dict
from algorithms.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *
import tensorflow as tf
import os
from common.tf_util import config_gpu


def ddpg_run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tf.config.optimizer.set_jit(True)
    config_gpu()

    env = RozumEnv()
    env = FrameSkip(env)
    env = FrameStack(env, 2, stack_key='pov')
    env = AccuracyLogWrapper(env, 10)
    env_dict, dtype_dict = get_dtype_dict(env)

    replay_buffer = PrioritizedReplayBuffer(size=50000, env_dict=env_dict)
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_keys = env.observation_space.spaces.keys()
        replay_buffer = DictWrapper(replay_buffer, state_prefix=('', 'next_', 'n_'),
                                    state_keys=state_keys)
    make_critic = get_network_builder("Critic_pov_angle")
    make_actor = get_network_builder("Actor_pov_angle")
    agent = DDPG(replay_buffer, make_critic, make_actor, env.observation_space, env.action_space, dtype_dict,
                 replay_start_size=100, train_quantity=100, train_freq=100, log_freq=20)
    summary_writer = tf.summary.create_file_writer('train/')
    with summary_writer.as_default():
        agent.train(env, 1000)
    env.close()
