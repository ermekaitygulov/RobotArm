import ray


import numpy as np
import tensorflow as tf
from tqdm import tqdm
import timeit

from algorithms.dqn import DQN


@ray.remote
class Learner(DQN):
    def __init__(self, remote_replay_buffer, parameter_server, build_model, update_target_net_mod=1000,
                 batch_size=32, replay_start_size=500, gamma=0.99, learning_rate=1e-4,
                 custom_loss=None):
        super().__init__(build_model, gamma, learning_rate, custom_loss)
        self.update_target_net_mod = update_target_net_mod
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        self.replay_buff = remote_replay_buffer
        self.parameter_server = parameter_server

    def update(self, steps, log_freq=10):
        progress = tqdm(total=steps)
        for i in range(1, steps + 1):
            progress.update(1)
            tree_idxes, minibatch, is_weights = self.replay_buff.sample(self.batch_size)

            pov = np.array([(np.array(data[0]) / 255) for data in minibatch], dtype='float32')
            action = np.array([data[1] for data in minibatch], dtype='int32')
            next_rewards = np.array([data[2] for data in minibatch], dtype='float32')
            next_pov = np.array([(np.array(data[3]) / 255) for data in minibatch], dtype='float32')
            done = np.array([data[4] for data in minibatch])
            n_pov = np.array([(np.array(data[5]) / 255) for data in minibatch], dtype='float32')
            n_reward = np.array([data[6] for data in minibatch], dtype='float32')
            n_done = np.array([data[7] for data in minibatch])
            actual_n = np.array([data[8] for data in minibatch], dtype='float32')
            gamma = np.array(self.gamma, dtype='float32')

            _, ntd_loss, _, _ = self.q_network_update(pov, action, next_rewards,
                                                      next_pov, done, n_pov,
                                                      n_reward, n_done, actual_n, is_weights, gamma)

            if tf.equal(self.optimizer.iterations % log_freq, 0):
                print("Epoch: ", self.optimizer.iterations.numpy())
                for key, metric in self.avg_metrics.items():
                    tf.summary.scalar(key, metric.result(), step=self.optimizer.iterations)
                    print('  {}:     {:.3f}'.format(key, metric.result()))
                    metric.reset_states()
                tf.summary.flush()
            self.replay_buff.batch_update(tree_idxes, ntd_loss)
        progress.close()

    def update_parameter_server(self):
        online_weights = self.online_model.get_weights()
        target_weights = self.target_model.get_weights()
        self.parameter_server.update_params.remote(online_weights, target_weights)


@ray.remote
class Actor(DQN):
    def __init__(self, thread_id, gamma, n_step, remote_replay_buffer, build_model, remote_param_server):
        super().__init__(list(), build_model, gamma, n_step)
        self.thread_id = thread_id
        self.remote_replay_buff = remote_replay_buffer
        self.parameter_server = remote_param_server

        if self.parameter_server:
            self.config = ray.get(self.parameter_server.get_config.remote())
            self.sync_nn_mod = self.config['sync_nn_mod']
            self.max_steps = self.config['max_steps']
            self.send_rollout_mod = self.config['send_rollout_mod']
            self.test_mod = self.config['test_mod']
            self.test_eps = self.config['test_eps']

        self.rollout = list()

    def train(self, env, epsilon=0.1, final_epsilon=0.01, eps_decay=0.99, **kwargs):
        max_reward, counter, threads_ep = - np.inf, 0, 0
        self.sync_with_param_server()
        done, score, state, start_time = False, 0, env.reset(), timeit.default_timer()
        global_step = ray.get(self.parameter_server.get_steps_done())
        while global_step < self.max_steps:
            action, q = self.choose_act(state, epsilon, env.sample_action)
            next_state, reward, done, _ = env.step(action)
            score += reward
            self.perceive([q, reward, next_state, done, False])
            counter += 1
            self.parameter_server.update_step.remote()
            global_step = ray.get(self.parameter_server.get_steps_done())
            state = next_state
            if counter % self.sync_nn_mod == 0:
                self.sync_with_param_server()
            if counter % self.send_rollout_mod == 0:
                priorities = self.priority_err(self.rollout)
                self.remote_replay_buff.receive.remote(self.rollout, priorities)
                self.replay_buff.clear()
            if done:
                epsilon = max(final_epsilon, epsilon * eps_decay)
                stop_time = timeit.default_timer()
                print("{}'s_episode: {}  score: {}  counter: {}  epsilon: {}  max: {}"
                      .format(self.thread_id, threads_ep, score, counter, epsilon, max_reward))
                print("RunTime: ", stop_time - start_time)
                tf.summary.scalar("reward", score, step=threads_ep)
                tf.summary.flush()
                done, score, state = False, 0, env.reset()
                start_time = timeit.default_timer()

    def validate(self, env):
        global_step = ray.get(self.parameter_server.get_steps_done.remote())
        while global_step < self.max_steps:
            if global_step % self.test_mod:
                self.sync_with_param_server()
                self.test(env, None, self.test_eps)
            global_step = ray.get(self.parameter_server.get_steps_done.remote())

    def priority_err(self, rollout):
        q_values = np.array([data[0] for data in rollout], dtype='float32')
        n_pov = np.array([(np.array(data[4]) / 255) for data in rollout], dtype='float32')
        n_reward = np.array([data[5] for data in rollout], dtype='float32')
        n_done = np.array([data[6] for data in rollout])
        actual_n = np.array([data[7] for data in rollout], dtype='float32')
        gamma = np.array(self.gamma, dtype='float32')
        is_weights = np.ones(len(rollout))

        ntd = self.td_loss(n_pov, q_values, n_done, n_reward, actual_n, gamma, is_weights)
        return ntd

    def sync_with_param_server(self):
        online_weights, target_weights = ray.get(self.parameter_server.return_params.remote())
        self.online_model.set_weights(online_weights)
        self.target_model.set_weights(target_weights)


@ray.remote
class ParameterServer(object):

    def __init__(self, config):
        self.online_params = None
        self.target_params = None
        self.steps_done = 0
        self.config = config

    def update_params(self, online_params, target_params):
        self.online_params = online_params
        self.target_params = target_params

    def return_params(self):
        return self.online_params, self.target_params

    def get_steps_done(self):
        return self.steps_done

    def update_step(self):
        self.steps_done += 1

    def get_config(self):
        return self.config
