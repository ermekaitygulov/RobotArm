import gym
import ray
from replay_buffers.util import DictWrapper, get_dtype_dict
from cpprb import ReplayBuffer

import numpy as np
import timeit

from algorithms.dqn.dqn import DQN


@ray.remote(num_gpus=0.5)
class Learner:
    def __init__(self, base=DQN, **kwargs):
        import tensorflow as tf
        from common.tf_util import config_gpu
        config_gpu()
        self.tf = tf
        self.tf.config.optimizer.set_jit(True)
        self.base = base(None, **kwargs)
        self.summary_writer = tf.summary.create_file_writer('train/learner/')

    def update_from_ds(self, ds, start_time, batch_size):
        with self.summary_writer.as_default():
            loss_list = list()
            indexes = ds.pop('indexes')
            ds = self.tf.data.Dataset.from_tensor_slices(ds)
            ds = ds.batch(batch_size)
            ds = ds.cache()
            ds = ds.prefetch(self.tf.data.experimental.AUTOTUNE)
            for batch in ds:
                priorities = self.base.nn_update(gamma=self.gamma, **batch)
                stop_time = timeit.default_timer()
                self.base.run_time_deque.append(stop_time - start_time)
                self.base.schedule()
                loss_list.append(priorities)
                start_time = timeit.default_timer()
        return indexes, np.concatenate(loss_list)

    @ray.method(num_return_vals=2)
    def get_weights(self):
        return self.base.get_online(), self.base.get_target()


@ray.remote(num_gpus=0, num_cpus=2)
class Actor:
    def __init__(self, base=DQN, thread_id=0, make_env=None, config_env=None, remote_counter=None,
                 rollout_size=300, **agent_kwargs):
        import tensorflow as tf
        self.tf = tf
        self.env = make_env('{}_thread'.format(thread_id), **config_env)
        env_dict, _ = get_dtype_dict(self.env)
        env_dict['q_value'] = {"dtype": "float32"}
        buffer = ReplayBuffer(size=rollout_size, env_dict=env_dict)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            state_keys = self.env.observation_space.spaces.keys()
            buffer = DictWrapper(buffer, state_prefix=('', 'next_', 'n_'), state_keys=state_keys)
        self.base = base(buffer, **agent_kwargs)
        self.summary_writer = self.tf.summary.create_file_writer('train/{}_actor/'.format(thread_id))
        self.max_reward = -np.inf
        self.env_state = None
        self.remote_counter = remote_counter

    def rollout(self, *weights):
        with self.summary_writer.as_default():
            self.base.set_weights(*weights)
            if self.env_state is None:
                done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
            else:
                done, score, state, start_time = self.env_state
            while self.base.replay_buff.get_stored_size() < self.base.replay_buff.get_buffer_size():
                action, q = self.base.choose_act(state, self.env.sample_action)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.base.perceive(state, action, reward, next_state, done, q_value=q)
                state = next_state
                if done:
                    global_ep = ray.get(self.remote_counter.increment.remote())
                    stop_time = timeit.default_timer()
                    if score > self.max_reward:
                        self.max_reward = score
                    print("episode: {}  score: {:.3f}  epsilon: {:.3f}  max: {:.3f}"
                          .format(global_ep-1, score, self.epsilon, self.max_reward))
                    print("RunTime: {:.3f}".format(stop_time - start_time))
                    self.tf.summary.scalar("reward", score, step=global_ep-1)
                    self.tf.summary.flush()
                    done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
            self.env_state = [done, score, state, start_time]
            rollout = self.base.replay_buff.get_all_transitions()
            priorities = self.priority_err(rollout)
            rollout.pop('q_value')
            self.base.replay_buff.clear()
            return rollout, priorities

    # TODO add validation method

    def priority_err(self, rollout):
        batch = {key: rollout[key] for key in ['q_value', 'n_done',
                                               'n_reward', 'actual_n', 'n_state']}
        for key in ['q_value', 'n_done', 'n_reward', 'actual_n']:
            batch[key] = np.squeeze(batch[key])
        n_target = self.base.compute_target(next_state=batch['n_state'],
                                       done=batch['n_done'],
                                       reward=batch['n_reward'],
                                       actual_n=batch['actual_n'],
                                       gamma=self.gamma)
        ntd = batch['q_value'] - n_target
        return np.abs(ntd)


@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_value(self):
        return self.value
