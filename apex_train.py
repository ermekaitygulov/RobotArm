import timeit
from argparse import ArgumentParser

import ray
import yaml

import algorithms
from algorithms.apex import Learner, Counter, Actor
from collections import defaultdict

from offpolicy_train import stack_env, make_discrete_env, make_continuous_env
from replay_buffers.util import DictWrapper, get_dtype_dict
from cpprb import PrioritizedReplayBuffer as cppPER
from nn_models.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *


def make_env(thread_id, n_actors=None, exploration_kwargs=None, env_kwargs=None, frame_stack=4, discretize=True):
    env_kwargs = env_kwargs if env_kwargs else {}
    exploration = apex_ranging(exploration_kwargs, thread_id, n_actors) if exploration_kwargs else {}
    environment = RozumEnv(**env_kwargs)
    if thread_id >= 0:
        environment = RozumLogWrapper(environment, 10, '{}_thread'.format(thread_id))
    if thread_id == -1:
        environment = RozumLogWrapper(environment, 10, 'Evaluate_thread')
    if frame_stack > 1:
        environment = stack_env(environment, frame_stack)
    if discretize:
        environment = make_discrete_env(environment, **exploration)
    else:
        environment = make_continuous_env(environment, **exploration)
    return environment


def apex_ranging(exploration, i, n_actors):
    assert isinstance(i, int)
    assert isinstance(exploration, dict)
    if i < 0:
        return {expl_name: np.array(expl_value) * 0 for expl_name, expl_value in exploration.items()}
    else:
        return {expl_name: np.array(expl_value) ** (1 + i / (n_actors - 1) * 0.7)
                for expl_name, expl_value in exploration.items()}


def make_remote_base(remote_config, n_actors):
    base = getattr(algorithms, remote_config['base'])

    def make_env_thunk(index):
        def thunk():
            return make_env(index, n_actors, **remote_config['env'])
        return thunk

    test_env = make_env_thunk(-2)()
    obs_space = test_env.observation_space
    action_space = test_env.action_space
    env_dict, dtype_dict = get_dtype_dict(test_env)
    test_env.close()

    remote_counter = Counter.remote()
    network_kwargs = dict()
    for arg_name, arg_value in remote_config['neural_network'].items():
        if isinstance(arg_value, dict):
            network_kwargs[arg_name] = get_network_builder(**arg_value)
        else:
            network_kwargs[arg_name] = get_network_builder(arg_value)
    dtype_dict['indexes'] = 'uint64'
    main_buffer = cppPER(env_dict=env_dict, **remote_config['buffer'])
    if isinstance(test_env.observation_space, gym.spaces.Dict):
        state_keys = test_env.observation_space.spaces.keys()
        main_buffer = DictWrapper(main_buffer, state_prefix=('', 'next_', 'n_'),
                                  state_keys=state_keys)
    remote_learner = Learner.remote(base=base, obs_space=obs_space, action_space=action_space,
                                    **remote_config['learner'], **remote_config['alg_args'], **network_kwargs)
    remote_actors = [Actor.remote(thread_id=i, base=base, make_env=make_env_thunk(i), remote_counter=remote_counter,
                                  obs_space=obs_space, action_space=action_space, **network_kwargs,
                                  **remote_config['actors'], **remote_config['alg_args']) for i in range(n_actors)]
    remote_evaluate = Actor.remote(thread_id='Evaluate', base=base, make_env=make_env_thunk(-1),
                                   remote_counter=remote_counter, obs_space=obs_space, action_space=action_space,
                                   wandb_group=remote_config['base'], **network_kwargs, **remote_config['actors'],
                                   **remote_config['alg_args'])
    return remote_learner, remote_actors, main_buffer, remote_counter, remote_evaluate


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', action='store', help='Path to config with params for chosen alg',
                        type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = defaultdict(dict)
        config.update(yaml.load(config_file, Loader=yaml.FullLoader))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
    os.environ["QT_DEBUG_PLUGINS"] = "0"
    ray.init(webui_host='0.0.0.0', num_gpus=1)

    # Preparation
    train_config = dict(max_eps=1000, replay_start_size=1000,
                        batch_size=128, sync_nn_mod=100, number_of_batchs=16,
                        beta=0.4, num_actors=4, rollout_size=70)
    if 'train' in config.keys():
        for key, value in config['train'].items():
            assert key in train_config.keys()
            train_config[key] = value
    learner, actors, replay_buffer, counter, evaluate = make_remote_base(config, train_config['num_actors'])

    # Start tasks
    online_weights, target_weights = learner.get_weights.remote()
    start_learner = False
    rollouts = {}
    for a in actors:
        rollouts[a.rollout.remote(train_config['rollout_size'], online_weights, target_weights)] = a
    rollouts[evaluate.test.remote(online_weights, target_weights)] = evaluate
    episodes_done = ray.get(counter.get_value.remote())
    optimization_step = 0
    priority_dict, ds = None, None

    # Main train process
    while episodes_done < train_config['max_eps']:
        ready_ids, _ = ray.wait(list(rollouts))
        first_id = ready_ids[0]
        first = rollouts.pop(first_id)
        if first == learner:
            optimization_step += 1
            start_time = timeit.default_timer()
            if optimization_step % train_config['sync_nn_mod'] == 0:
                online_weights, target_weights = first.get_weights.remote()
            rollouts[first.update_from_ds.remote(ds, start_time, train_config['batch_size'])] = first
            indexes, priorities = ray.get(first_id)
            indexes = indexes.copy()
            priorities = priorities.copy()
            replay_buffer.update_priorities(indexes=indexes, priorities=priorities)
            ds = replay_buffer.sample(train_config['number_of_batchs'] * train_config['batch_size'],
                                      train_config['beta'])
        elif first == evaluate:
            rollouts[evaluate.test.remote(online_weights, target_weights)] = evaluate
        else:
            rollouts[first.rollout.remote(train_config['rollout_size'], online_weights, target_weights)] = first
            data, priorities = ray.get(first_id)
            priorities = priorities.copy()
            replay_buffer.add(priorities=priorities, **data)
        if replay_buffer.get_stored_size() > train_config['replay_start_size'] and not start_learner:
            start_time = timeit.default_timer()
            ds = replay_buffer.sample(train_config['number_of_batchs'] * train_config['batch_size'],
                                      train_config['beta'])
            rollouts[learner.update_from_ds.remote(ds, start_time, train_config['batch_size'])] = learner
            ds = replay_buffer.sample(train_config['number_of_batchs'] * train_config['batch_size'],
                                      train_config['beta'])
            start_learner = True
        episodes_done = ray.get(counter.get_value.remote())
    ray.timeline()
