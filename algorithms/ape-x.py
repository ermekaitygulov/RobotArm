import random
import ray

from collections import deque

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import timeit

from utils.util import take_vector_elements


@ray.remote
class Learner:
    def __init__(self, remote_replay_buffer, parameter_server, build_model, update_target_net_mod=1000,
                 batch_size=32, replay_start_size=500, gamma=0.99, learning_rate=1e-4,
                 custom_loss=None):
        # global
        self.update_target_net_mod = update_target_net_mod
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.gamma = gamma

        self.replay_buff = remote_replay_buffer
        self.parameter_server = parameter_server
        self.online_model = build_model('Online')
        self.target_model = build_model('Target')
        self.custom_loss = custom_loss

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.avg_metrics = dict()

    def update(self, steps, log_freq=10):
        progress = tqdm(total=steps)
        for i in range(1, steps + 1):
            progress.update(1)
            tree_idxes, minibatch, is_weights = self.replay_buff.sample(self.batch_size)

            pov_batch = np.array([(np.array(data[0])/255).astype('float32') for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch], dtype='int32')
            reward_batch = np.array([data[2] for data in minibatch], dtype='float32')
            next_pov_batch = np.array([(np.array(data[3])/255).astype('float32') for data in minibatch])
            done_batch = np.array([data[4] for data in minibatch])
            n_pov_batch = np.array([(np.array(data[6])/255).astype('float32') for data in minibatch])
            n_reward = np.array([data[7] for data in minibatch], dtype='float32')
            n_done = np.array([data[8] for data in minibatch])
            actual_n = np.array([data[9] for data in minibatch], dtype='float32')
            gamma = np.array(self.gamma, dtype='float32')

            abs_loss = self.q_network_update(pov_batch, action_batch, reward_batch,
                                             next_pov_batch, done_batch, n_pov_batch,
                                             n_reward, n_done, actual_n, is_weights, gamma)

            if tf.equal(self.optimizer.iterations % log_freq, 0):
                print("Epoch: ", self.optimizer.iterations.numpy())
                for key, metric in self.avg_metrics.items():
                    tf.summary.scalar(key, metric.result(), step=self.optimizer.iterations)
                    print('  {}:     {:.3f}'.format(key, metric.result()))
                    metric.reset_states()
                tf.summary.flush()
            self.replay_buff.batch_update(tree_idxes, abs_loss)
        progress.close()

    @tf.function
    def q_network_update(self, pov_batch, action_batch, reward_batch,
                         next_pov_batch, done_batch, n_pov_batch,
                         n_reward, n_done, actual_n, is_weights, gamma):
        online_variables = self.online_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(online_variables)
            huber = tf.keras.losses.Huber(0.4)
            q_value = self.online_model(pov_batch, training=True)
            q_value = take_vector_elements(q_value, action_batch)
            target = self.compute_target(next_pov_batch, done_batch, reward_batch, 1, gamma)
            target = tf.stop_gradient(target)
            td_loss = huber(target, q_value, is_weights)
            print('----------------- creating metrics ---------------')
            if 'TD' not in self.avg_metrics:
                self.avg_metrics['TD'] = tf.keras.metrics.Mean(name='TD', dtype=tf.float32)
            self.avg_metrics['TD'].update_state(td_loss)


            abs_loss = tf.abs(target - q_value)

            n_target = self.compute_target(n_pov_batch, n_done, n_reward, actual_n, gamma)
            ntd_loss = huber(n_target, q_value, is_weights)
            if 'nTD' not in self.avg_metrics:
                self.avg_metrics['nTD'] = tf.keras.metrics.Mean(name='nTD', dtype=tf.float32)
            self.avg_metrics['nTD'].update_state(ntd_loss)

            l2 = tf.add_n(self.online_model.losses)
            if 'l2' not in self.avg_metrics:
                self.avg_metrics['l2'] = tf.keras.metrics.Mean(name='l2', dtype=tf.float32)
            self.avg_metrics['l2'].update_state(l2)

            all_losses = td_loss + ntd_loss + l2
            if 'all_losses' not in self.avg_metrics:
                self.avg_metrics['all_losses'] = tf.keras.metrics.Mean(name='all_losses', dtype=tf.float32)
            self.avg_metrics['all_losses'].update_state(all_losses)

        gradients = tape.gradient(all_losses, online_variables)
        self.optimizer.apply_gradients(zip(gradients, online_variables))
        return abs_loss, td_loss, ntd_loss, l2, all_losses

    def compute_target(self, next_pov, done, reward, actual_n, gamma):
        q_network = self.online_model(next_pov, training=True)
        argmax_actions = tf.argmax(q_network, axis=1, output_type='int32')
        q_target = self.target_model(next_pov, training=True)
        target = take_vector_elements(q_target, argmax_actions)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward
        return target

    def update_parameter_server(self):
        weights = self.online_model.get_weights()
        self.parameter_server.update_params.remote(weights)

    def target_update(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def save(self, out_dir=None):
        self.online_model.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_model.load_weights(out_dir)


@ray.remote()
class Actor:
    def __init__(self, thread_id, remote_replay_buffer, build_model, remote_param_server):
        # global
        self.thread_id = thread_id
        self.remote_replay_buff = remote_replay_buffer
        self.parameter_server = remote_param_server
        self.online_model = build_model('Online')
        self.target_model = build_model('Target')

        if self.parameter_server:
            self.config = ray.get(self.parameter_server.get_config.remote())
            self.gamma = self.config['gamma']
            self.n_deque = deque([], maxlen=self.config['n_step'])
            self.sync_nn_mod = self.config['sync_nn_mod']
            self.max_steps = self.config['max_steps']
            self.send_rollout_mod = self.config['send_rollout_mod']
            self.test_mod = self.config['test_mod']
            self.test_eps = self.config['test_eps']

        self.rollout = list()

    def train(self, env, epsilon=0.1, final_epsilon=0.01, eps_decay=0.99):
        max_reward = - np.inf
        counter = 0
        threads_ep = 0
        self.sync_with_param_server()
        done, score, state = False, 0, env.reset()
        start_time = timeit.default_timer()
        global_step = ray.get(self.parameter_server.get_steps_done())
        while global_step < self.max_steps:
            if random.random() > epsilon:
                action = self.choose_act(state)
            else:
                action = env.sample_action()
            next_state, reward, done, _ = env.step(action)
            score += reward
            self.perceive([state, action, reward, next_state, done, False])
            counter += 1
            self.parameter_server.update_steps()
            global_step = ray.get(self.parameter_server.get_steps_done())
            state = next_state
            if counter % self.sync_nn_mod == 0:
                self.sync_with_param_server()
            if counter % self.send_rollout_mod == 0:
                self.remote_replay_buff.receive.remote(self.rollout)
                self.rollout.clear()
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

    def test(self, env, name, number_of_trials=1):
        """
        Method for testing model in environment
        :param env:
        :param name:
        :param number_of_trials:
        :return:
        """
        # restore POV agent's graph
        if name:
            self.load(name)

        total_reward = 0

        for trial_index in range(number_of_trials):
            reward = 0
            done = False
            observation = env.reset()
            while not done:
                action = self.choose_act(observation)
                observation, r, done, _ = env.step(action)
                reward += r
            total_reward += reward
            print("reward/avg_reward for {} trial: {}; {}".format(trial_index, reward,
                                                                  total_reward / (trial_index + 1)))
        env.reset()
        return total_reward

    def perceive(self, transition):
        self.n_deque.append(transition)
        if len(self.n_deque) == self.n_deque.maxlen or transition[4]:
            while len(self.n_deque) != 0:
                n_step_pov = self.n_deque[-1][3]
                n_step_done = self.n_deque[-1][4]
                n_step_r = sum([t[2] * self.gamma ** (i + 1) for i, t in enumerate(self.n_deque)])
                self.n_deque[0].append(n_step_pov)
                self.n_deque[0].append(n_step_r)
                self.n_deque[0].append(n_step_done)
                self.n_deque[0].append(len(self.n_deque) + 1)
                self.rollout.append(self.n_deque.popleft())
                if not n_step_done:
                    break

    def choose_act(self, state):
        inputs = (np.array(state) / 255).astype('float32')
        q_values = self.online_model(inputs[None], training=False)[0]
        return np.argmax(q_values).astype('int32')

    def sync_with_param_server(self):
        weights = ray.get(self.parameter_server.return_params.remote())
        self.online_model.set_weights(weights)

    def save(self, out_dir=None):
        self.online_model.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_model.load_weights(out_dir)


@ray.remote
class ParameterServer(object):

    def __init__(self, config):
        self.params = []
        self.steps_done = 0
        self.config = config

    def update_params(self, new_params):
        self.params = new_params

    def return_params(self):
        return self.params

    def get_steps_done(self):
        return self.steps_done

    def update_step(self):
        self.steps_done += 1

    def get_config(self):
        return self.config
