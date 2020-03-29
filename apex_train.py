import ray

from algorithms.apex import Learner, Counter, Actor
from replay_buffers.apex_buffer import ApeXBuffer
from algorithms.model import ClassicCnn, DuelingModel
from environments.pyrep_env import RozumEnv
from utils.wrappers import *
import os
import time


def make_env(name):
    env = RozumEnv()
    env = FrameSkip(env)
    env = FrameStack(env, 2)
    env = AccuracyLogWrapper(env, 10, name)
    discrete_dict = dict()
    robot_dof = env.action_space.shape[0]
    for i in range(robot_dof):
        discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
        discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
    env = DiscreteWrapper(env, discrete_dict)
    return env


def make_model(name, input_shape, output_shape):
    import tensorflow as tf
    from utils.util import config_gpu
    config_gpu()
    base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])
    head = DuelingModel([1024], output_shape)
    model = tf.keras.Sequential([base, head], name)
    model.build((None, ) + input_shape)
    return model


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ray.init(webui_host='0.0.0.0', num_gpus=1)
    n_actors = 3
    max_eps = 1e+6
    replay_start_size = 1500
    batch_size = 128
    sync_nn_mod = 300
    dtype_dict = {'state': 'float32',
                  'action': 'int32',
                  'reward': 'float32',
                  'next_state': 'float32',
                  'done': 'bool',
                  'n_state': 'float32',
                  'n_reward': 'float32',
                  'n_done': 'bool',
                  'actual_n': 'float32'}

    test_env = make_env('test_name')
    obs_shape = test_env.observation_space.shape
    action_shape = test_env.action_space.n
    test_env.close()

    counter = Counter.remote()
    replay_buffer = ApeXBuffer.remote(int(1e5))
    learner = Learner.remote(replay_buffer, make_model, obs_shape, action_shape, update_target_nn_mod=1000,
                             gamma=0.99, learning_rate=1e-4)
    actors = [Actor.remote(i, make_model, obs_shape, action_shape, make_env, counter, gamma=0.99, n_step=5)
              for i in range(n_actors)]
    online_weights, target_weights = learner.get_weights.remote()

    @ray.remote
    def remote_sleep():
        while ray.get(replay_buffer.len.remote()) < replay_start_size:
            time.sleep(60)

    rollouts = {}
    for a in actors:
        rollouts[a.rollout.remote(online_weights, target_weights, rollout_size=300)] = a
    rollouts[remote_sleep.remote()] = 'learner_waiter'
    episodes_done = ray.get(counter.get_value.remote())
    tree_ids = None
    optimization_step = 0
    while episodes_done < max_eps:
        ready_ids, _ = ray.wait(list(rollouts))
        first_id = ready_ids[0]
        first = rollouts.pop(first_id)
        if first == 'learner_waiter':
            tree_ids, minibatch, is_weights = replay_buffer.sample.remote(batch_size, dtype_dict)
            rollouts[learner.update_asynch.remote(minibatch, is_weights, log_freq=100)] = learner
        elif first == learner:
            optimization_step += 1
            ntd = first_id
            replay_buffer.batch_update.remote(tree_ids, ntd)
            tree_ids, minibatch, is_weights = replay_buffer.sample.remote(batch_size, dtype_dict)
            if optimization_step % sync_nn_mod == 0:
                online_weights, target_weights = first.get_weights.remote()
            rollouts[first.update_asynch.remote(minibatch, is_weights, log_freq=100)] = first
        else:
            replay_buffer.receive.remote(first_id)
            rollouts[first.rollout.remote(online_weights, target_weights, rollout_size=300)] = first
        episodes_done = ray.get(counter.get_value.remote())
    ray.timeline()
