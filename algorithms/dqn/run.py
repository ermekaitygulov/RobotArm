import yaml

from algorithms.dqn.dqn import DQN
from cpprb import PrioritizedReplayBuffer as cppPER
from replay_buffers.stable_baselines import PrioritizedReplayBuffer
from replay_buffers.util import DictWrapper, get_dtype_dict
from algorithms.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *
import tensorflow as tf
import os
from common.tf_util import config_gpu


def make_env(epsilon=0.1, final_epsilon=0.01, epsilon_decay=0.99, **env_kwargs):
    env = RozumEnv(**env_kwargs)
    env = RozumLogWrapper(env, 10)
    discrete_dict = dict()
    robot_dof = env.action_space.shape[0] - 1
    for i in range(robot_dof):
        # joint actions
        discrete_angle = 5 / 180
        discrete_dict[i] = [discrete_angle if j == i
                            else 0 for j in range(robot_dof)] + [1., ]
        discrete_dict[i + robot_dof] = [-discrete_angle if j == i
                                        else 0 for j in range(robot_dof)] + [1., ]
    # gripper action
    discrete_dict[2 * robot_dof] = [0., ] * (robot_dof + 1)
    env = DiscreteWrapper(env, discrete_dict)
    env = EpsilonExploration(env, epsilon, final_epsilon, epsilon_decay)
    return env


def dqn_run(config_path):
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
    make_model = get_network_builder(config['neural_network'])
    agent = DQN(make_model, env.observation_space, env.action_space, replay_buff=replay_buffer,
                dtype_dict=dtype_dict, **config['agent'])

    if 'pretrain_weights' in config:
        agent.load(**config['pretrain_weights'])
    train_config = config['train']
    summary_writer = tf.summary.create_file_writer(train_config.pop('log_dir'))
    with summary_writer.as_default():
        agent.train(env, **train_config)
    env.close()
