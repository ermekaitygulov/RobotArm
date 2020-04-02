from collections import deque

import ray


import numpy as np
import timeit

from algorithms.dqn import DQN


@ray.remote(num_gpus=0.3)
class Learner(DQN):
    def __init__(self, build_model, obs_shape, action_shape, update_target_nn_mod=1000,
                 gamma=0.99, learning_rate=1e-4, log_freq=100):
        import tensorflow as tf
        super().__init__(None, build_model, obs_shape, action_shape,
                         gamma=gamma, learning_rate=learning_rate, update_target_nn_mod=update_target_nn_mod,
                         log_freq=log_freq)
        self.summary_writer = tf.summary.create_file_writer('train/learner/')

    def update_asynch(self, minibatch, start_time):
        with self.summary_writer.as_default():
            casted_batch = {key: minibatch[key].astype(self.dtype_dict[key]) for key in self.dtype_dict.keys()
                            if key != 'weights'}
            casted_batch['state'] = (casted_batch['state'] / 255).astype('float32')
            casted_batch['next_state'] = (casted_batch['next_state'] / 255).astype('float32')
            casted_batch['n_state'] = (casted_batch['n_state'] / 255).astype('float32')
            _, ntd_loss, _, _ = self.q_network_update(casted_batch['state'], casted_batch['action'],
                                                      casted_batch['reward'], casted_batch['next_state'],
                                                      casted_batch['done'], casted_batch['n_state'],
                                                      casted_batch['n_reward'], casted_batch['n_done'],
                                                      casted_batch['actual_n'], casted_batch['weights'], self.gamma)

            stop_time = timeit.default_timer()
            self._run_time_deque.append(1/(stop_time - start_time))
            self.schedule()
            return ntd_loss

    @ray.method(num_return_vals=2)
    def get_weights(self):
        return self.online_model.get_weights(), self.target_model.get_weights()


@ray.remote(num_gpus=0, num_cpus=2)
class Actor(DQN):
    def __init__(self, thread_id, build_model, obs_shape, action_shape,
                 make_env, remote_counter, gamma=0.99, n_step=10):
        import tensorflow as tf
        self.env = make_env('{}_thread'.format(thread_id))
        super().__init__(list(), build_model, obs_shape, action_shape,
                         gamma=gamma, n_step=n_step)
        self.summary_writer = tf.summary.create_file_writer('train/{}_actor/'.format(thread_id))
        self.epsilon = 0.1
        self.epsilon_decay = 0.99
        self.final_epsilon = 0.01
        self.max_reward = -np.inf
        self.env_state = None
        self.remote_counter = remote_counter

    def rollout(self, online_weights, target_weights, rollout_size=300):
        import tensorflow as tf
        with self.summary_writer.as_default():
            self.online_model.set_weights(online_weights)
            self.target_model.set_weights(target_weights)
            if self.env_state is None:
                done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
            else:
                done, score, state, start_time = self.env_state
            while len(self.replay_buff) < rollout_size:
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
                    tf.summary.scalar("reward", score, step=global_ep-1)
                    tf.summary.flush()
                    done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
                    self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
            self.env_state = [done, score, state, start_time]
            priorities = self.priority_err(self.replay_buff)
            rollout = self.replay_buff.copy()
            self.replay_buff.clear()
            return rollout, priorities

    def validate(self, test_mod=100, test_eps=10, max_eps=1e+6):
        import tensorflow as tf
        with self.summary_writer.as_default():
            global_ep = ray.get(self.parameter_server.get_eps_done.remote())
            while global_ep < max_eps:
                if (global_ep + 1) % test_mod == 0:
                    self.sync_with_param_server()
                    total_reward = self.test(self.env, None, test_eps, False)
                    total_reward /= test_eps
                    print("validation_reward (mean): {}".format(total_reward))
                    tf.summary.scalar("validation", total_reward, step=global_ep)
                    tf.summary.flush()
                global_ep = ray.get(self.parameter_server.get_eps_done.remote())

    def priority_err(self, rollout):
        q_values = np.array([[data['q_value']] for data in rollout], dtype='float32')
        n_state = np.array([(np.array(data['n_state'])/255) for data in rollout], dtype='float32')
        n_reward = np.array([data['n_reward'] for data in rollout], dtype='float32')
        n_done = np.array([data['n_done'] for data in rollout])
        actual_n = np.array([data['actual_n'] for data in rollout], dtype='float32')

        ntd = self.td_loss(n_state, q_values, n_done, n_reward, actual_n, self.gamma)
        return ntd.numpy()


@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_value(self):
        return self.value

