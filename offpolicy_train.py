import yaml
import algorithms
from nn_models.model import get_network_builder
from argparse import ArgumentParser
import tensorflow as tf

from common.tf_util import config_gpu
from common.wrappers import *
from environments.pyrep_env import RozumEnv
from replay_buffers.util import get_dtype_dict, DictWrapper
from cpprb import PrioritizedReplayBuffer as cppPER
from replay_buffers.stable_baselines import PrioritizedReplayBuffer
import os


def make_env(exploration_kwargs=None, env_kwargs=None, frame_stack=4, stack_keys=None, discretize=True):
    env_kwargs = env_kwargs if env_kwargs else {}
    exploration_kwargs = exploration_kwargs if exploration_kwargs else {}
    environment = RozumEnv(**env_kwargs)
    environment = RozumLogWrapper(environment, 10)
    if frame_stack > 1:
        stack_keys = stack_keys if stack_keys else list(environment.observation_space.spaces.keys())
        environment = stack_env(environment, frame_stack, stack_keys)
    if discretize:
        environment = make_discrete_env(environment, **exploration_kwargs)
    else:
        environment = make_continuous_env(environment, **exploration_kwargs)
    return environment


def make_continuous_env(environment, mu=0., sigma_epsilon=(0.1, 0.1)):
    mu = np.ones_like(environment.action_space.low) * mu
    sigma = np.ones_like(environment.action_space.low)
    sigma[:-1] *= sigma_epsilon[0]
    sigma[-1] *= 0
    environment = UncorrelatedExploration(environment, mu, sigma)
    environment = E3exploration(environment, sigma_epsilon[1])
    return environment


def make_discrete_env(environment, epsilon=0.1, final_epsilon=0.01, epsilon_decay=0.99):
    discrete_dict = dict()
    robot_dof = environment.action_space.shape[0] - 1
    for joint in range(robot_dof):
        # joint actions
        discrete_angle = 5 / 180
        discrete_dict[joint] = [discrete_angle if j == joint
                                else 0 for j in range(robot_dof)] + [1., ]
        discrete_dict[joint + robot_dof] = [-discrete_angle if j == joint
                                            else 0 for j in range(robot_dof)] + [1., ]
    # gripper action
    discrete_dict[2 * robot_dof] = [0., ] * (robot_dof + 1)
    environment = DiscreteWrapper(environment, discrete_dict)
    environment = EpsilonExploration(environment, epsilon, final_epsilon, epsilon_decay)
    return environment


def stack_env(environment, k, stack_keys):
    if isinstance(environment.observation_space, gym.spaces.Dict):
        for obs_key in environment.observation_space.spaces.keys():
            if obs_key in stack_keys:
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

    summary_writer = tf.summary.create_file_writer(config.pop('log_dir'))
    with summary_writer.as_default():
        if 'train' in config:
            train_config = config['train']
            agent.train(env, **train_config)
        if 'test' in config:
            test_config = config['test']
            agent.test(env, **test_config)
        env.reset()
        env.close()
