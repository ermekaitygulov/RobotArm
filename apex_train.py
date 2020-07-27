import timeit
from argparse import ArgumentParser

import ray
import yaml

import algorithms
from algorithms.apex import Learner, Counter, Actor

from offpolicy_train import stack_env, make_discrete_env, make_continuous_env
from replay_buffers.stable_baselines import PrioritizedReplayBuffer
from replay_buffers.util import DictWrapper, get_dtype_dict
from cpprb import PrioritizedReplayBuffer as cppPER
from nn_models.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *
import wandb
from collections import defaultdict


def make_env(thread_id, n_actors=None, exploration_kwargs=None, env_kwargs=None, frame_stack=4, discretize=True,
             wandb_group_id=None):
    env_kwargs = env_kwargs if env_kwargs else {}
    expl_values = apex_ranging(exploration_kwargs, thread_id, n_actors) if exploration_kwargs else {}
    environment = RozumEnv(**env_kwargs)
    if thread_id >= 0:
        environment = RozumLogWrapper(environment, 100, '{}_thread'.format(thread_id))
    if thread_id == -1:
        environment = RozumLogWrapper(environment, 100, 'Evaluate_thread', wandb_group_id=wandb_group_id)
    if frame_stack > 1:
        environment = stack_env(environment, frame_stack)
    if discretize:
        environment = make_discrete_env(environment, **expl_values)
    else:
        environment = make_continuous_env(environment, **expl_values)
    return environment


def apex_ranging(exploration_kwargs, actor_id, n_actors):
    assert isinstance(actor_id, int)
    assert isinstance(exploration_kwargs, dict)
    if actor_id < 0:
        return {expl_name: np.array(expl_value) * 0 for expl_name, expl_value in exploration_kwargs.items()}
    else:
        return {expl_name: np.array(expl_value) ** (1 + actor_id / (n_actors - 1) * 0.7)
                for expl_name, expl_value in exploration_kwargs.items()}


def make_remote_base(apex_config, env_dict=None, dtype_dict=None, obs_space=None, action_space=None):
    filler_config = defaultdict(dict)
    filler_config.update(apex_config)
    base = getattr(algorithms, apex_config['base'])
    n_actors = apex_config['num_actors']

    def make_env_thunk(index):
        def thunk():
            return make_env(index, n_actors, **filler_config['env'])
        return thunk

    if obs_space is None or action_space is None:
        test_env = make_env_thunk(-2)()
        obs_space = test_env.observation_space
        action_space = test_env.action_space
        test_env.close()
    if env_dict is None or dtype_dict is None:
        env_dict, dtype_dict = get_dtype_dict(obs_space, action_space)

    remote_counter = Counter.remote()
    network_kwargs = dict()
    for arg_name, arg_value in filler_config['neural_network'].items():
        if isinstance(arg_value, dict):
            network_kwargs[arg_name] = get_network_builder(**arg_value)
        else:
            network_kwargs[arg_name] = get_network_builder(arg_value)
    if 'cpp' in filler_config['buffer'].keys() and filler_config['buffer'].pop('cpp'):
        dtype_dict['indexes'] = 'uint64'
        main_buffer = cppPER(env_dict=env_dict, **filler_config['buffer'])
    else:
        main_buffer = PrioritizedReplayBuffer(env_dict=env_dict, **filler_config['buffer'])
    if isinstance(obs_space, gym.spaces.Dict):
        state_keys = obs_space.spaces.keys()
        main_buffer = DictWrapper(main_buffer, state_prefix=('', 'next_', 'n_'),
                                  state_keys=state_keys)
    if 'learner_resource' in filler_config:
        remote_learner = Learner.options(**filler_config['learner_resource']).remote(base=base,
                                                                                     obs_space=obs_space,
                                                                                     action_space=action_space,
                                                                                     **filler_config['learner'],
                                                                                     **filler_config['alg_args'],
                                                                                     **network_kwargs)
    else:
        remote_learner = Learner.remote(base=base, obs_space=obs_space, action_space=action_space,
                                        **filler_config['learner'], **filler_config['alg_args'], **network_kwargs)
    if 'actor_resource' in filler_config:
        remote_actors = [Actor.options(**filler_config['actor_resource']).remote(thread_id=actor_id, base=base,
                                                                                 make_env=make_env_thunk(actor_id),
                                                                                 remote_counter=remote_counter,
                                                                                 obs_space=obs_space,
                                                                                 action_space=action_space,
                                                                                 env_dict=env_dict,
                                                                                 **network_kwargs,
                                                                                 **filler_config['actors'],
                                                                                 **filler_config['alg_args'])
                         for actor_id in range(n_actors)]
        remote_evaluate = Actor.options(**filler_config['actor_resource']).remote(thread_id='Evaluate', base=base,
                                                                                  make_env=make_env_thunk(-1),
                                                                                  remote_counter=remote_counter,
                                                                                  obs_space=obs_space,
                                                                                  action_space=action_space,
                                                                                  **network_kwargs,
                                                                                  **filler_config['actors'],
                                                                                  **filler_config['alg_args'])
    else:
        remote_actors = [Actor.remote(thread_id=actor_id, base=base, make_env=make_env_thunk(actor_id),
                                      remote_counter=remote_counter, obs_space=obs_space, action_space=action_space,
                                      **network_kwargs, **filler_config['actors'], **filler_config['alg_args'])
                         for actor_id in range(n_actors)]
        remote_evaluate = Actor.remote(thread_id='Evaluate', base=base, make_env=make_env_thunk(-1),
                                       remote_counter=remote_counter, obs_space=obs_space, action_space=action_space,
                                       **network_kwargs, **filler_config['actors'], **filler_config['alg_args'])
    return remote_learner, remote_actors, main_buffer, remote_counter, remote_evaluate


def apex(remote_learner, remote_actors, main_buffer, remote_counter, remote_evaluate,
         log_wandb=False, max_eps=1000, replay_start_size=1000, batch_size=128,
         sync_nn_mod=100, number_of_batchs=16, beta=0.4, rollout_size=70,
         save_nn_mod=100, save_dir='train'):
    # Start tasks
    online_weights, target_weights = remote_learner.get_weights.remote()
    start_learner = False
    rollouts = {}
    for a in remote_actors:
        rollouts[a.rollout.remote(rollout_size, online_weights, target_weights)] = a
    rollouts[remote_evaluate.test.remote(save_dir, online_weights, target_weights)] = remote_evaluate
    episodes_done = ray.get(remote_counter.get_value.remote())
    optimization_step = 0
    priority_dict, ds = None, None

    # Main train process
    while episodes_done < max_eps:
        ready_ids, _ = ray.wait(list(rollouts))
        first_id = ready_ids[0]
        first = rollouts.pop(first_id)
        if first == remote_learner:
            optimization_step += 1
            start_time = timeit.default_timer()
            if optimization_step % sync_nn_mod == 0:
                save = (optimization_step % save_nn_mod == 0) * save_dir
                online_weights, target_weights = first.get_weights.remote(save)
            rollouts[first.update_from_ds.remote(ds, start_time, batch_size)] = first
            indexes, priorities = ray.get(first_id)
            indexes = indexes.copy()
            priorities = priorities.copy()
            main_buffer.update_priorities(indexes=indexes, priorities=priorities)
            ds = main_buffer.sample(number_of_batchs * batch_size, beta)
        elif first == remote_evaluate:
            score, eps_number = ray.get(first_id)
            if log_wandb:
                wandb.log({'Score': score, 'episode': eps_number})
            rollouts[remote_evaluate.test.remote(save_dir, online_weights, target_weights)] = remote_evaluate
        else:
            rollouts[first.rollout.remote(rollout_size, online_weights, target_weights)] = first
            data, priorities = ray.get(first_id)
            priorities = priorities.copy()
            main_buffer.add(priorities=priorities, **data)
        if main_buffer.get_stored_size() > replay_start_size and not start_learner:
            start_time = timeit.default_timer()
            ds = main_buffer.sample(number_of_batchs * batch_size, beta)
            rollouts[remote_learner.update_from_ds.remote(ds, start_time, batch_size)] = remote_learner
            ds = main_buffer.sample(number_of_batchs * batch_size, beta)
            start_learner = True
        episodes_done = ray.get(remote_counter.get_value.remote())
    ray.timeline()


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

    learner, actors, replay_buffer, counter, evaluate = make_remote_base(config)
    apex(learner, actors, replay_buffer, counter, evaluate, args.wandb, **config['train'])
