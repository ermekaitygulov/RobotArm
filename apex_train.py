import timeit

import ray

from algorithms.apex import Learner, Counter, Actor
from replay_buffers.apex_buffer import ApeXBuffer
from algorithms.model import ClassicCnn, DuelingModel, MLP
from environments.pyrep_env import RozumEnv
from utils.wrappers import *
import os
import time


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


def make_model(name, obs_space, action_space):
    import tensorflow as tf
    from utils.util import config_gpu
    config_gpu()
    pov = tf.keras.Input(shape=obs_space['pov'].shape)
    angles = tf.keras.Input(shape=obs_space['angles'].shape)
    pov_base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])(pov)
    angles_base = MLP([512, 256])(angles)
    base = tf.keras.layers.concatenate([pov_base, angles_base])
    head = DuelingModel([1024], action_space.n)(base)
    model = tf.keras.Model(inputs={'pov': pov, 'angles': angles}, outputs=head, name=name)
    return model


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ray.init(webui_host='0.0.0.0', num_gpus=1)
    n_actors = 3
    max_eps = 1000
    replay_start_size = 1000
    batch_size = 128
    sync_nn_mod = 100
    rollout_size = 100
    number_of_batchs = 16

    test_env = make_env('test_name')
    obs_space = test_env.observation_space
    action_space = test_env.action_space
    test_env.close()
    env_dict = {'action': {'dtype': 'int32'},
                'reward': {'dtype': 'float32'},
                'done': {'dtype': 'bool'},
                'n_reward': {'dtype': 'float32'},
                'n_done': {'dtype': 'bool'},
                'actual_n': {'dtype': 'float32'},
                }
    for prefix in ('', 'next_', 'n_'):
        env_dict[prefix + 'pov'] = {'shape': obs_space['pov'].shape,
                                    'dtype': 'uint8'}
        env_dict[prefix + 'angles'] = {'shape': obs_space['angles'].shape,
                                       'dtype': 'float32'}

    counter = Counter.remote()
    rb_class = ray.remote(ApeXBuffer)
    replay_buffer = rb_class.remote(size=int(1e5), env_dict=env_dict,
                                    state_prefix=('', 'next_', 'n_'), state_keys=('pov', 'angles',))
    learner = Learner.remote(make_model, obs_space, action_space, update_target_nn_mod=1000,
                             gamma=0.9, learning_rate=1e-4, log_freq=100)
    actors = [Actor.remote(i, make_model, obs_space, action_space, make_env, counter,
                           buffer_size=rollout_size, gamma=0.99, n_step=5) for i in range(n_actors)]
    online_weights, target_weights = learner.get_weights.remote()

    @ray.remote
    def remote_sleep():
        while ray.get(replay_buffer.get_stored_size.remote()) < replay_start_size:
            time.sleep(60)

    rollouts = {}
    for a in actors:
        rollouts[a.rollout.remote(online_weights, target_weights, rollout_size)] = a
    rollouts[remote_sleep.remote()] = 'learner_waiter'
    episodes_done = ray.get(counter.get_value.remote())
    ready_tree_ids, ds, proc_tree_ids = None, None, None
    optimization_step = 0
    while episodes_done < max_eps:
        ready_ids, _ = ray.wait(list(rollouts))
        first_id = ready_ids[0]
        first = rollouts.pop(first_id)
        if first == 'learner_waiter':
            ds = replay_buffer.sample.remote(number_of_batchs*batch_size)
            start_time = timeit.default_timer()
            rollouts[learner.update_from_ds.remote(ds, start_time, batch_size)] = learner
            ds = replay_buffer.sample.remote(number_of_batchs*batch_size)
        elif first == learner:
            optimization_step += 1
            start_time = timeit.default_timer()
            priority_dict = first_id
            replay_buffer.update_priorities.remote(priority_dict)
            if optimization_step % sync_nn_mod == 0:
                online_weights, target_weights = first.get_weights.remote()
            rollouts[first.update_from_ds.remote(ds, start_time, batch_size)] = first
            ds = replay_buffer.sample.remote(number_of_batchs * batch_size)
        else:
            replay_buffer.add.remote(first_id)
            rollouts[first.rollout.remote(online_weights, target_weights, rollout_size)] = first
        episodes_done = ray.get(counter.get_value.remote())
    ray.timeline()
