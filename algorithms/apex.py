import os
from collections import deque

import gym
import ray

from common.tf_util import config_gpu
from replay_buffers.util import DictWrapper, get_dtype_dict
from cpprb import ReplayBuffer

import numpy as np
import timeit

from algorithms.dqn import DoubleDuelingDQN


@ray.remote(num_gpus=0.5)
class Learner:
    def __init__(self, base=DoubleDuelingDQN, pretrain_weights=None, **actor_kwargs):
        import tensorflow as tf
        from common.tf_util import config_gpu
        config_gpu()
        self.tf = tf
        self.tf.config.optimizer.set_jit(True)
        self.base = base(replay_buff=None, **actor_kwargs)
        if pretrain_weights:
            self.load(**pretrain_weights)
        self.summary_writer = tf.summary.create_file_writer('train/Learner_logger/')

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.base, name)

    def update_from_ds(self, ds, start_time, batch_size):
        with self.summary_writer.as_default():
            loss_list = list()
            indexes = ds.pop('indexes')
            self.histogram(ds['state'], 'cube', -6)
            self.histogram(ds['state'], 'arm', -17, -11, lambda x: x / np.pi * 180)
            ds = self.tf.data.Dataset.from_tensor_slices(ds)
            ds = ds.batch(batch_size)
            ds = ds.cache()
            ds = ds.prefetch(self.tf.data.experimental.AUTOTUNE)
            for batch in ds:
                priorities = self.nn_update(gamma=self.gamma, **batch)
                stop_time = timeit.default_timer()
                self.run_time_deque.append(stop_time - start_time)
                self.schedule()
                loss_list.append(priorities)
                start_time = timeit.default_timer()
        return indexes, np.concatenate(loss_list)

    def histogram(self, minibatch, key, left=None, right=None, operation=None):
        data = minibatch[key][:, left:right]
        if operation:
            data = operation(data)
        for dim in range(data.shape[-1]):
            self.tf.summary.histogram('{}/{}_{}'.format(key, dim, key), data[:, dim], step=self.q_optimizer.iterations)

    @ray.method(num_return_vals=2)
    def get_weights(self, save_dir=None):
        if save_dir:
            self.save(save_dir)
        return self.get_online(), self.get_target()

    def set_weights(self, *weights):
        self.base.set_weights(*weights)


@ray.remote(num_gpus=0.1, num_cpus=2)
class Actor:
    def __init__(self, base=DoubleDuelingDQN, thread_id=0, make_env=None, remote_counter=None,
                 avg_window=10, pretrain_weights=None, **agent_kwargs):
        import tensorflow as tf
        config_gpu()
        self.tf = tf
        self.thread_id = thread_id
        self.env = make_env()
        self.base = base(replay_buff=self._init_buff(1), **agent_kwargs)
        if pretrain_weights:
            self.load(**pretrain_weights)
        self.env_state = None
        self.remote_counter = remote_counter
        self.local_ep = 0
        self.avg_reward = deque([], maxlen=avg_window)
        self.summary_writer = self.tf.summary.create_file_writer('train/{}_actor/'.format(thread_id))
        self._max_reward = -np.inf

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.base, name)

    def _init_buff(self, size):
        env_dict, _ = get_dtype_dict(self.env.observation_space, self.env.action_space)
        env_dict['q_value'] = {"dtype": "float32"}
        buffer = ReplayBuffer(size=size, env_dict=env_dict)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            state_keys = self.env.observation_space.spaces.keys()
            buffer = DictWrapper(buffer, state_prefix=('', 'next_', 'n_'), state_keys=state_keys)
        return buffer

    def rollout(self, rollout_size, *weights):
        if rollout_size != self.replay_buff.get_buffer_size():
            self.base.replay_buff = self._init_buff(rollout_size)
        if weights:
            self.set_weights(*weights)
        with self.summary_writer.as_default():
            if self.env_state is None:
                done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
            else:
                done, score, state, start_time = self.env_state
            while self.replay_buff.get_stored_size() < self.replay_buff.get_buffer_size():
                action, q = self.choose_act(state, self.env.sample_action)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.perceive(state, action, reward, next_state, done, q_value=q)
                state = next_state
                if done:
                    self.tf.summary.scalar('Score', score, step=self.local_ep)
                    self.tf.summary.flush()
                    self.local_ep += 1
                    self.avg_reward.append(score)
                    avg = sum(self.avg_reward)/len(self.avg_reward)
                    global_ep, max_reward = ray.get(self.remote_counter.increment.remote(score))
                    stop_time = timeit.default_timer()
                    print("episode: {}  score: {:.3f}  max: {:.3f}  {}_avg: {:.3f}"
                          .format(global_ep-1, score, max_reward, self.thread_id, avg))
                    print("RunTime: {:.3f}".format(stop_time - start_time))
                    print("{} transitions collected".format(self.replay_buff.get_stored_size()))
                    done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
            self.env_state = [done, score, state, start_time]
            rollout = self.replay_buff.get_all_transitions()
            priorities = self.priority_err(rollout)
            rollout.pop('q_value')
            self.replay_buff.clear()
            return rollout, priorities

    def priority_err(self, rollout):
        batch = {key: rollout[key] for key in ['q_value', 'n_done',
                                               'n_reward', 'actual_n', 'n_state']}
        for key in ['q_value', 'n_done', 'n_reward', 'actual_n']:
            batch[key] = np.squeeze(batch[key])
        n_target = self.compute_target(next_state=batch['n_state'],
                                       done=batch['n_done'],
                                       reward=batch['n_reward'],
                                       actual_n=batch['actual_n'],
                                       gamma=self.gamma)
        ntd = batch['q_value'] - n_target
        return np.abs(ntd)

    def test(self, save_dir=None, *weights):
        if weights:
            self.set_weights(*weights)
        with self.summary_writer.as_default():
            done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
            while not done:
                action, q = self.choose_act(state, self.env.sample_action)
                state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    self.tf.summary.scalar('Score', score, step=self.local_ep)
                    self.tf.summary.flush()
                    self.avg_reward.append(score)
                    avg = sum(self.avg_reward)/len(self.avg_reward)
                    if avg > self._max_reward:
                        self._max_reward = avg
                        if save_dir:
                            self.save(out_dir=os.path.join(save_dir, 'max'))
                    stop_time = timeit.default_timer()
                    print("Evaluate episode: {}  score: {:.3f}  avg: {:.3f}  max_avg: {:.3f}"
                          .format(self.local_ep, score, avg, self._max_reward))
                    print("RunTime: {:.3f}".format(stop_time - start_time))
                    self.local_ep += 1
        return score, self.local_ep


@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0
        self.max_reward = -np.inf

    def increment(self, reward):
        if reward >= self.max_reward:
            self.max_reward = reward
        self.value += 1
        return self.value, self.max_reward

    def get_value(self):
        return self.value
