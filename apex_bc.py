import os
from argparse import ArgumentParser

import gym
import ray
import yaml

import algorithms
from apex_train import make_remote_base, apex
import wandb
import tensorflow as tf

from common.data_loader import DataLoader
from common.tf_util import config_gpu
from nn_models.model import get_network_builder
from cpprb import DQfDBuffer
from replay_buffers.util import get_dtype_dict, DictWrapper

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', action='store', help='Path to config with params for chosen alg',
                        type=str, required=True)
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
    os.environ["QT_DEBUG_PLUGINS"] = "0"
    ray.init(webui_host='0.0.0.0', num_gpus=1)
    if args.wandb:
        group_id = config['base'] + '_' + str(wandb.util.generate_id())
        wandb = wandb.init(anonymous='allow', project="Rozum", group=group_id)
        wandb.config.update(config)
        config['env']['wandb_group_id'] = group_id
    tf.config.optimizer.set_jit(True)
    config_gpu()
    data_loader = DataLoader(**config['data_loader'])
    env_dict, dtype_dict = get_dtype_dict(data_loader.observation_space, data_loader.action_space)
    env_dict.update(demo={'dtype': 'float32'}), dtype_dict.update(demo='float32')
    replay_buffer = DQfDBuffer(env_dict=env_dict, **config['buffer'])
    if isinstance(data_loader.observation_space, gym.spaces.Dict):
        state_keys = data_loader.observation_space.spaces.keys()
        replay_buffer = DictWrapper(replay_buffer, state_prefix=('', 'next_', 'n_'),
                                    state_keys=state_keys)
    network_kwargs = dict()
    for key, value in config['neural_network'].items():
        if isinstance(value, dict):
            network_kwargs[key] = get_network_builder(**value)
        else:
            network_kwargs[key] = get_network_builder(value)

    base = getattr(algorithms, config['base'])
    agent = base(obs_space=data_loader.observation_space, action_space=data_loader.action_space,
                 replay_buff=replay_buffer, dtype_dict=dtype_dict,
                 **config['alg_args'], **network_kwargs)
    if 'pretrain_weights' in config:
        agent.load(**config['pretrain_weights'])
    agent.add_demo(data_loader)
    pretrain_config = config['pretrain']
    summary_writer = tf.summary.create_file_writer(pretrain_config['log_dir'])
    with summary_writer.as_default():
        agent.update(pretrain_config['steps'])
        if 'save_path' in pretrain_config:
            agent.save(pretrain_config['save_path'])
    online_weights, target_weights = agent.get_online(), agent.get_target()
    learner, actors, _, counter, evaluate = make_remote_base(config, env_dict, dtype_dict,
                                                             data_loader.observation_space,
                                                             data_loader.action_space)
    learner.set_weights.remote(online_weights, target_weights)
    apex(learner, actors, agent.replay_buff, counter, evaluate, args.wandb, **config['train'])
