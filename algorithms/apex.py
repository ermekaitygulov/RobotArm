import ray
from replay_buffers.cpprb_wrapper import RB

import numpy as np
import timeit

from algorithms.dqn import DQN


@ray.remote(num_gpus=0.5)
class Learner(DQN):
    def __init__(self, build_model, obs_shape, action_space, update_target_nn_mod=1000,
                 gamma=0.99, learning_rate=1e-4, log_freq=100):
        import tensorflow as tf
        tf.config.optimizer.set_jit(True)
        super().__init__(None, build_model, obs_shape, action_space,
                         gamma=gamma, learning_rate=learning_rate, update_target_nn_mod=update_target_nn_mod,
                         log_freq=log_freq)
        self.summary_writer = tf.summary.create_file_writer('train/learner/')

    def update_from_ds(self, ds, start_time, batch_size):
        import tensorflow as tf

        loss_list = list()
        indexes = ds.pop('indexes')
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.batch(batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        for batch in ds:
            _, ntd_loss, _, _ = self.q_network_update(gamma=self.gamma, **batch)
            stop_time = timeit.default_timer()
            self._run_time_deque.append(stop_time - start_time)
            self.schedule()
            loss_list.append(ntd_loss)
            start_time = timeit.default_timer()
        return indexes, np.abs(np.concatenate(loss_list))

    @ray.method(num_return_vals=2)
    def get_weights(self):
        return self.online_model.get_weights(), self.target_model.get_weights()


@ray.remote(num_gpus=0, num_cpus=2)
class Actor(DQN):
    def __init__(self, thread_id, build_model, obs_space, action_space,
                 make_env, remote_counter, buffer_size, gamma=0.99, n_step=10):
        import tensorflow as tf
        self.env = make_env('{}_thread'.format(thread_id))
        env_dict = {'action': {'dtype': 'int32'},
                    'reward': {'dtype': 'float32'},
                    'done': {'dtype': 'bool'},
                    'n_reward': {'dtype': 'float32'},
                    'n_done': {'dtype': 'bool'},
                    'actual_n': {'dtype': 'float32'},
                    'q_value': {'dtype': 'float32'}
                    }
        for prefix in ('', 'next_', 'n_'):
            env_dict[prefix + 'pov'] = {'shape': obs_space['pov'].shape,
                                        'dtype': 'uint8'}
            env_dict[prefix + 'angles'] = {'shape': obs_space['angles'].shape,
                                           'dtype': 'float32'}
        buffer = RB(size=buffer_size, env_dict=env_dict,
                    state_prefix=('', 'next_', 'n_'), state_keys=('pov', 'angles',))
        super().__init__(buffer, build_model, obs_space, action_space,
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
            while self.replay_buff.get_stored_size() < rollout_size:
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
            rollout = self.replay_buff.get_all_transitions()
            priorities = self.priority_err(rollout)
            rollout.pop('q_value')
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
