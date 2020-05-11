import gym
import ray
from replay_buffers.util import DictWrapper, get_dtype_dict
from cpprb import ReplayBuffer

import numpy as np
import timeit

from algorithms.dqn.dqn import DQN


@ray.remote(num_gpus=0.5)
class Learner(DQN):
    def __init__(self, build_model, obs_shape, action_space, update_target_nn_mod=1000,
                 gamma=0.99, learning_rate=1e-4, log_freq=100):
        import tensorflow as tf
        self.tf = tf
        self.tf.config.optimizer.set_jit(True)
        super().__init__(None, build_model, obs_shape, action_space,
                         gamma=gamma, learning_rate=learning_rate, update_target_nn_mod=update_target_nn_mod,
                         log_freq=log_freq)
        self.summary_writer = tf.summary.create_file_writer('train/learner/')

    def update_from_ds(self, ds, start_time, batch_size):
        loss_list = list()
        indexes = ds.pop('indexes')
        ds = self.tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.batch(batch_size)
        ds = ds.cache()
        ds = ds.prefetch(self.tf.data.experimental.AUTOTUNE)
        for batch in ds:
            priorities = self.q_network_update(gamma=self.gamma, **batch)
            stop_time = timeit.default_timer()
            self._run_time_deque.append(stop_time - start_time)
            self.schedule()
            loss_list.append(priorities)
            start_time = timeit.default_timer()
        return indexes, np.concatenate(loss_list)

    @ray.method(num_return_vals=2)
    def get_weights(self):
        return self.online_model.get_weights(), self.target_model.get_weights()


@ray.remote(num_gpus=0, num_cpus=2)
class Actor(DQN):
    def __init__(self, thread_id, build_model, obs_space, action_space,
                 make_env, config_env, remote_counter, rollout_size, gamma=0.99, n_step=10):
        import tensorflow as tf
        self.tf = tf
        self.env = make_env('{}_thread'.format(thread_id), **config_env)
        env_dict, _ = get_dtype_dict(self.env)
        env_dict['q_value'] = {"dtype": "float32"}
        buffer = ReplayBuffer(size=rollout_size, env_dict=env_dict)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            state_keys = self.env.observation_space.spaces.keys()
            buffer = DictWrapper(buffer, state_prefix=('', 'next_', 'n_'), state_keys=state_keys)
        super().__init__(buffer, build_model, obs_space, action_space,
                         gamma=gamma, n_step=n_step)
        self.summary_writer = self.tf.summary.create_file_writer('train/{}_actor/'.format(thread_id))
        self.epsilon = 0.1
        self.epsilon_decay = 0.99
        self.final_epsilon = 0.01
        self.max_reward = -np.inf
        self.env_state = None
        self.remote_counter = remote_counter

    def rollout(self, online_weights, target_weights):
        with self.summary_writer.as_default():
            self.online_model.set_weights(online_weights)
            self.target_model.set_weights(target_weights)
            if self.env_state is None:
                done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
            else:
                done, score, state, start_time = self.env_state
            while self.replay_buff.get_stored_size() < self.replay_buff.get_buffer_size():
                action, q = self.choose_act(state, self.epsilon, self.env.sample_action)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.perceive(state, action, reward, next_state, done, q_value=q)
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
                    self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
            self.env_state = [done, score, state, start_time]
            rollout = self.replay_buff.get_all_transitions()
            priorities = self.priority_err(rollout)
            rollout.pop('q_value')
            self.replay_buff.clear()
            return rollout, priorities

    def validate(self, test_mod=100, test_eps=10, max_eps=1e+6):
        with self.summary_writer.as_default():
            global_ep = ray.get(self.parameter_server.get_eps_done.remote())
            while global_ep < max_eps:
                if (global_ep + 1) % test_mod == 0:
                    self.sync_with_param_server()
                    total_reward = self.test(self.env, None, test_eps, False)
                    total_reward /= test_eps
                    print("validation_reward (mean): {}".format(total_reward))
                    self.tf.summary.scalar("validation", total_reward, step=global_ep)
                    self.tf.summary.flush()
                global_ep = ray.get(self.parameter_server.get_eps_done.remote())

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


@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_value(self):
        return self.value
