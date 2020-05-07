import timeit
import ray
from algorithms.apex.apex import Learner, Counter, Actor
from replay_buffers.util import DictWrapper, get_dtype_dict
from cpprb import PrioritizedReplayBuffer
from algorithms.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *


def make_env(name):
    env = RozumEnv()
    env = FrameSkip(env)
    env = FrameStack(env, 2, stack_key='pov')
    env = AccuracyLogWrapper(env, 10, name)
    discrete_dict = dict()
    robot_dof = env.action_space.shape[0]
    for i in range(robot_dof):
        discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
        discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
    env = DiscreteWrapper(env, discrete_dict)
    return env


def apex_run():
    ray.init(webui_host='0.0.0.0', num_gpus=1)
    n_actors = 3
    max_eps = 1000
    replay_start_size = 1000
    batch_size = 128
    sync_nn_mod = 100
    rollout_size = 100
    number_of_batchs = 16
    buffer_size = int(1e5)

    test_env = make_env('test_name')
    obs_space = test_env.observation_space
    action_space = test_env.action_space
    env_dict, dtype_dict = get_dtype_dict(test_env)
    test_env.close()

    counter = Counter.remote()
    make_model = get_network_builder("DuelingDQN_pov_angle")
    replay_buffer = PrioritizedReplayBuffer(size=buffer_size, env_dict=env_dict)
    replay_buffer = DictWrapper(replay_buffer, state_prefix=('', 'next_', 'n_'),
                                state_keys=('pov', 'angles',))
    learner = Learner.remote(make_model, obs_space, action_space, update_target_nn_mod=1000,
                             gamma=0.9, learning_rate=1e-4, log_freq=100)
    actors = [Actor.remote(i, make_model, obs_space, action_space, make_env, counter,
                           buffer_size=rollout_size, gamma=0.99, n_step=5) for i in range(n_actors)]
    online_weights, target_weights = learner.get_weights.remote()
    start_learner = False
    rollouts = {}
    for a in actors:
        rollouts[a.rollout.remote(online_weights, target_weights, rollout_size)] = a
    episodes_done = ray.get(counter.get_value.remote())
    optimization_step = 0
    priority_dict, ds = None, None
    while episodes_done < max_eps:
        ready_ids, _ = ray.wait(list(rollouts))
        first_id = ready_ids[0]
        first = rollouts.pop(first_id)
        if first == learner:
            optimization_step += 1
            start_time = timeit.default_timer()
            if optimization_step % sync_nn_mod == 0:
                online_weights, target_weights = first.get_weights.remote()
            rollouts[first.update_from_ds.remote(ds, start_time, batch_size)] = first
            indexes, priorities = ray.get(first_id)
            indexes = indexes.copy()
            priorities = priorities.copy()
            replay_buffer.update_priorities(indexes=indexes, priorities=priorities)
            ds = replay_buffer.sample(number_of_batchs * batch_size)
        else:
            rollouts[first.rollout.remote(online_weights, target_weights, rollout_size)] = first
            data, priorities = ray.get(first_id)
            priorities = priorities.copy()
            replay_buffer.add(priorities=priorities, **data)
        if replay_buffer.get_stored_size() > replay_start_size and not start_learner:
            start_time = timeit.default_timer()
            ds = replay_buffer.sample(number_of_batchs * batch_size)
            rollouts[learner.update_from_ds.remote(ds, start_time, batch_size)] = learner
            ds = replay_buffer.sample(number_of_batchs * batch_size)
            start_learner = True
        episodes_done = ray.get(counter.get_value.remote())
    ray.timeline()


if __name__ == '__main__':
    apex_run()
