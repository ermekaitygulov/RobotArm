import gym

from offpolicy_train import make_env
import yaml
import algorithms
from nn_models.model import get_network_builder
from argparse import ArgumentParser
import tensorflow as tf

from common.tf_util import config_gpu
from replay_buffers.util import get_dtype_dict, DictWrapper
from replay_buffers.bc_buffer import DQfDBuffer
import os


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
    env_dict, dtype_dict = get_dtype_dict(env)
    replay_buffer = DQfDBuffer(env_dict=env_dict, **config['buffer'])
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
    summary_writer = tf.summary.create_file_writer(train_config.pop('log_dir'))
    with summary_writer.as_default():
        agent.train(env, **train_config)
    env.reset()
    env.close()