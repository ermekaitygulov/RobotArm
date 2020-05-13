import timeit
import ray
import yaml

from algorithms.apex.apex import Learner, Counter, Actor
from replay_buffers.util import DictWrapper, get_dtype_dict
from cpprb import PrioritizedReplayBuffer as cppPER
from replay_buffers.stable_baselines import PrioritizedReplayBuffer
from algorithms.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *
from common.tf_util import config_gpu


def make_env(name, obs_space_keys=('pov', 'arm'), frame_skip=4, frame_stack=4):
    env = RozumEnv(obs_space_keys)
    if frame_skip > 1:
        env = FrameSkip(env, frame_skip)
    if frame_stack > 1 and 'pov' in obs_space_keys:
        env = FrameStack(env, frame_stack, stack_key='pov')
    env = AccuracyLogWrapper(env, 10, name)
    discrete_dict = dict()
    robot_dof = env.action_space.shape[0] - 1
    for i in range(robot_dof):
        # joint actions
        discrete_angle = 5 / 180
        discrete_dict[i] = [discrete_angle
                            if j == i else 0 for j in range(robot_dof)] + [1., ]
        discrete_dict[i + robot_dof] = [-discrete_angle
                                        if j == i else 0 for j in range(robot_dof)] + [1., ]
    # gripper action
    discrete_dict[2 * robot_dof] = [0., ] * (robot_dof + 1)
    env = DiscreteWrapper(env, discrete_dict)
    return env


def apex_run(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    ray.init(webui_host='0.0.0.0', num_gpus=1)
    try:
        n_actors = config['actors'].pop('num')
    except KeyError:
        n_actors = 1

    test_env = make_env('test_name', **config['env'])
    obs_space = test_env.observation_space
    action_space = test_env.action_space
    env_dict, dtype_dict = get_dtype_dict(test_env)
    test_env.close()

    counter = Counter.remote()
    make_model = get_network_builder(config['neural_network'])
    if 'cpp' in config['buffer'].keys() and config['buffer'].pop('cpp'):
        dtype_dict['indexes'] = 'uint64'
        replay_buffer = cppPER(env_dict=env_dict, **config['buffer'])
    else:
        replay_buffer = PrioritizedReplayBuffer(env_dict=env_dict, **config['buffer'])
    if isinstance(test_env.observation_space, gym.spaces.Dict):
        state_keys = test_env.observation_space.spaces.keys()
        replay_buffer = DictWrapper(replay_buffer, state_prefix=('', 'next_', 'n_'),
                                    state_keys=state_keys)
    learner = Learner.remote(make_model, obs_space, action_space, **config['learner'])
    actors = [Actor.remote(i, make_model, obs_space, action_space, make_env, config['env'], counter,
                           epsilon=0.4**(1+i/(n_actors-1)*0.7),
                           **config['actors']) for i in range(n_actors)]
    online_weights, target_weights = learner.get_weights.remote()
    start_learner = False
    rollouts = {}
    for a in actors:
        rollouts[a.rollout.remote(online_weights, target_weights)] = a
    episodes_done = ray.get(counter.get_value.remote())
    optimization_step = 0
    priority_dict, ds = None, None

    train_config = dict(max_eps=1000, replay_start_size=1000,
                        batch_size=128, sync_nn_mod=100, number_of_batchs=16)
    if 'train' in config.keys():
        for key, value in config['train'].items():
            assert key in train_config.keys()
            train_config[key] = value

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
            ds = replay_buffer.sample(train_config['number_of_batchs'] * train_config['batch_size'])
        else:
            rollouts[first.rollout.remote(online_weights, target_weights)] = first
            data, priorities = ray.get(first_id)
            priorities = priorities.copy()
            replay_buffer.add(priorities=priorities, **data)
        if replay_buffer.get_stored_size() > train_config['replay_start_size'] and not start_learner:
            start_time = timeit.default_timer()
            ds = replay_buffer.sample(train_config['number_of_batchs'] * train_config['batch_size'])
            rollouts[learner.update_from_ds.remote(ds, start_time, train_config['batch_size'])] = learner
            ds = replay_buffer.sample(train_config['number_of_batchs'] * train_config['batch_size'])
            start_learner = True
        episodes_done = ray.get(counter.get_value.remote())
    ray.timeline()
