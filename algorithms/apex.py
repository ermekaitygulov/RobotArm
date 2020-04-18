import ray


import numpy as np
import timeit

from algorithms.dqn import DQN
from utils.nested_dict import dict_op, dict_append


@ray.remote(num_gpus=0.3)
class Learner(DQN):
    def __init__(self, build_model, obs_shape, action_shape, update_target_nn_mod=1000,
                 gamma=0.99, learning_rate=1e-4, log_freq=100):
        import tensorflow as tf
        tf.config.optimizer.set_jit(True)
        super().__init__(None, build_model, obs_shape, action_shape,
                         gamma=gamma, learning_rate=learning_rate, update_target_nn_mod=update_target_nn_mod,
                         log_freq=log_freq)
        self.summary_writer = tf.summary.create_file_writer('train/learner/')

    def update_from_ds(self, ds, start_time, batch_size):
        import tensorflow as tf
        loss_list = list()
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.map(self.preprocess_ds)
        ds = ds.batch(batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        for batch in ds:
            _, ntd_loss, _, _ = self.q_network_update(gamma=self.gamma, **batch)
            stop_time = timeit.default_timer()
            self._run_time_deque.append(stop_time - start_time)
            self.schedule()
            loss_list.append(np.abs(ntd_loss))
            start_time = timeit.default_timer()
        return np.concatenate(loss_list)

    @ray.method(num_return_vals=2)
    def get_weights(self):
        return self.online_model.get_weights(), self.target_model.get_weights()


@ray.remote(num_gpus=0, num_cpus=2)
class Actor(DQN):
    dtype_dict = {'state': {'pov': 'float32', 'angles': 'float32'},
                  'action': 'int32',
                  'reward': 'float32',
                  'next_state': {'pov': 'float32', 'angles': 'float32'},
                  'done': 'bool',
                  'n_state': {'pov': 'float32', 'angles': 'float32'},
                  'n_reward': 'float32',
                  'n_done': 'bool',
                  'actual_n': 'float32',
                  'weights': 'float32',
                  'q_value': 'float32'}

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
        import tensorflow as tf
        # TODO remove tf ds
        batch_keys = ['n_state', 'q_value', 'n_done', 'n_reward', 'actual_n']
        ignore_keys = [key for key in rollout[0].keys() if key not in batch_keys]
        ds = self._encode_rollout(rollout, ignore_keys)
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.map(self.preprocess_ds)
        ds = ds.batch(self.batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        priorities = list()
        for batch in ds:
            ntd = self.td_loss(batch['n_state'],
                               batch['q_value'],
                               batch['n_done'],
                               batch['n_reward'],
                               batch['actual_n'],
                               self.gamma)
            priorities.append(ntd)
        return np.abs(np.concatenate(priorities))

    @staticmethod
    def _encode_rollout(rollout, ignore_keys):
        batch = dict_op(rollout[0], lambda _: list(), ignore_keys)
        for b in rollout:
            data = dict_op(b, np.array, ignore_keys)
            batch = dict_append(batch, data)
        batch = dict_op(batch, np.array)
        return batch


@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_value(self):
        return self.value
