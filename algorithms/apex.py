import ray


import numpy as np
import timeit

from algorithms.dqn import DQN


@ray.remote(num_gpus=0.3)
class Learner(DQN):
    def __init__(self, remote_replay_buffer, build_model, obs_shape, action_shape,
                 parameter_server, update_target_net_mod=1000, gamma=0.99, learning_rate=1e-4,
                 batch_size=32, replay_start_size=1000):
        import tensorflow as tf
        super().__init__(remote_replay_buffer, build_model, obs_shape, action_shape,
                         gamma=gamma, learning_rate=learning_rate, update_target_net_mod=update_target_net_mod,
                         batch_size=batch_size, replay_start_size=replay_start_size)
        self.parameter_server = parameter_server
        self.summary_writer = tf.summary.create_file_writer('train/learner/')

    def update(self, max_eps=10000, log_freq=100, **kwargs):
        import tensorflow as tf
        self.update_parameter_server()
        while ray.get(self.replay_buff.len.remote()) < self.replay_start_size:
            continue
        with self.summary_writer.as_default():
            global_eps = ray.get(self.parameter_server.get_eps_done.remote())
            start_time = timeit.default_timer()
            while global_eps < max_eps:
                tree_idxes, minibatch, is_weights = ray.get(self.replay_buff.sample.remote(self.batch_size))

                state = (minibatch['state'] / 255).astype('float32')
                action = (minibatch['action']).astype('int32')
                next_rewards = (minibatch['reward']).astype('float32')
                next_state = (minibatch['next_state'] / 255).astype('float32')
                done = minibatch['done']
                n_state = (minibatch['n_state'] / 255).astype('float32')
                n_reward = (minibatch['n_reward']).astype('float32')
                n_done = (minibatch['n_done'])
                actual_n = (minibatch['actual_n']).astype('float32')
                is_weights = is_weights.astype('float32')

                _, ntd_loss, _, _ = self.q_network_update(state, action, next_rewards,
                                                          next_state, done, n_state,
                                                          n_reward, n_done, actual_n, is_weights, self.gamma)

                if tf.equal(self.optimizer.iterations % self.update_target_net_mod, 0):
                    self.target_update()
                if tf.equal(self.optimizer.iterations % log_freq, 0):
                    stop_time = timeit.default_timer()
                    runtime = stop_time - start_time
                    start_time = stop_time
                    print("LearnerEpoch({}it/sec): ".format(log_freq / runtime), self.optimizer.iterations.numpy())
                    for key, metric in self.avg_metrics.items():
                        tf.summary.scalar(key, metric.result(), step=self.optimizer.iterations)
                        print('  {}:     {:.3f}'.format(key, metric.result()))
                        metric.reset_states()
                    tf.summary.flush()
                self.replay_buff.batch_update.remote(tree_idxes, ntd_loss)
                global_eps = ray.get(self.parameter_server.get_eps_done.remote())
                self.update_parameter_server()

    def update_parameter_server(self):
        online_weights = self.online_model.get_weights()
        target_weights = self.target_model.get_weights()
        self.parameter_server.update_params.remote(online_weights, target_weights)


@ray.remote
class Actor(DQN):
    def __init__(self, thread_id, remote_replay_buffer, build_model, obs_shape, action_shape,
                 make_env, remote_param_server, gamma=0.99, n_step=10,
                 sync_nn_steps=100, send_rollout_steps=64, test=False):
        import tensorflow as tf
        self.env = make_env('{}_thread'.format(thread_id), test)
        super().__init__(list(), build_model, obs_shape, action_shape,
                         gamma=gamma, n_step=n_step)
        self.remote_replay_buff = remote_replay_buffer
        self.parameter_server = remote_param_server
        self.sync_nn_shedule = {'next': sync_nn_steps, 'every': sync_nn_steps}
        self.send_rollout_schedule = {'next': send_rollout_steps, 'every': send_rollout_steps}
        self.summary_writer = tf.summary.create_file_writer('train/{}_actor/'.format(thread_id))

    def train(self, epsilon=0.1, final_epsilon=0.01, eps_decay=0.99,
              max_eps=1e+6, **kwargs):
        import tensorflow as tf
        with self.summary_writer.as_default():
            max_reward, counter = - np.inf, 0
            self.sync_with_param_server()
            done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
            global_ep = ray.get(self.parameter_server.get_eps_done.remote())
            while global_ep < max_eps:
                action, q = self.choose_act(state, epsilon, self.env.sample_action)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.perceive(state, action, reward, next_state, done, q_value=q)
                counter += 1
                state = next_state
                self.schedule(counter)
                if done:
                    self.parameter_server.update_eps.remote()
                    global_ep = ray.get(self.parameter_server.get_eps_done.remote())
                    stop_time = timeit.default_timer()
                    print("episode: {}  score: {}  counter: {}  epsilon: {}  max: {}"
                          .format(global_ep-1, score, counter, epsilon, max_reward))
                    print("RunTime: ", stop_time - start_time)
                    tf.summary.scalar("reward", score, step=global_ep-1)
                    tf.summary.flush()
                    done, score, state, start_time = False, 0, self.env.reset(), timeit.default_timer()
                    epsilon = max(final_epsilon, epsilon * eps_decay)

    def schedule(self, counter):
        if counter > self.sync_nn_shedule['next'] >= 0:
            self.sync_nn_shedule['next'] += self.sync_nn_shedule['every']
            self.sync_with_param_server()
        if counter > self.send_rollout_schedule['next'] >= 0:
            self.send_rollout_schedule['next'] += self.send_rollout_schedule['every']
            priorities = self.priority_err(self.replay_buff)
            self.remote_replay_buff.receive.remote(self.replay_buff, priorities)
            self.replay_buff.clear()

    def validate(self, test_mod=100, test_eps=10, max_eps=1e+6):
        import tensorflow as tf
        with self.summary_writer.as_default():
            global_ep = ray.get(self.parameter_server.get_eps_done.remote())
            while global_ep < max_eps:
                if (global_ep + 1) % test_mod == 0:
                    self.sync_with_param_server()
                    total_reward = self.test(self.env, None, test_eps)
                    total_reward /= test_eps
                    tf.summary.scalar("validation", total_reward, step=global_ep)
                    tf.summary.flush()
                global_ep = ray.get(self.parameter_server.get_eps_done.remote())

    def priority_err(self, rollout):
        q_values = np.array([data['q_value'] for data in rollout], dtype='float32')
        n_state = np.array([(np.array(data['n_state'])/255) for data in rollout], dtype='float32')
        n_reward = np.array([data['n_reward'] for data in rollout], dtype='float32')
        n_done = np.array([data['n_done'] for data in rollout])
        actual_n = np.array([data['actual_n'] for data in rollout], dtype='float32')

        ntd = self.td_loss(n_state, q_values, n_done, n_reward, actual_n, self.gamma)
        return ntd

    def sync_with_param_server(self):
        while ray.get(self.parameter_server.return_params.remote())[0] is None:
            continue
        online_weights, target_weights = ray.get(self.parameter_server.return_params.remote())
        self.online_model.set_weights(online_weights)
        self.target_model.set_weights(target_weights)


@ray.remote
class ParameterServer(object):

    def __init__(self):
        self.online_params = None
        self.target_params = None
        self.eps_done = 0

    def update_params(self, online_params, target_params):
        self.online_params = online_params
        self.target_params = target_params

    def return_params(self):
        return self.online_params, self.target_params

    def get_eps_done(self):
        return self.eps_done

    def update_eps(self):
        self.eps_done += 1
