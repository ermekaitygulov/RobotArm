import random

from collections import deque

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import timeit

from utils.util import take_vector_elements


class DQNbase:
    def __init__(self, build_model, gamma=0.99, learning_rate=1e-4,
                 custom_loss=None):
        # global
        self.gamma = gamma
        self.online_model = build_model('Online')
        self.target_model = build_model('Target')
        self.custom_loss = custom_loss
        self.online_variables = self.online_model.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.huber_loss = tf.keras.losses.Huber(0.4, tf.keras.losses.Reduction.NONE)
        self.avg_metrics = dict()

    @tf.function
    def q_network_update(self, q_values, nex_reward, next_pov, done, n_pov,
                         n_reward, n_done, actual_n, is_weights, gamma):

        with tf.GradientTape() as tape:
            tape.watch(self.online_variables)
            td_loss = self.td_loss(next_pov, q_values, done, nex_reward, 1, gamma, is_weights)
            mean_td = tf.reduce_mean(td_loss)
            self.update_metrics('TD', mean_td)

            ntd_loss = self.td_loss(n_pov, q_values, n_done, n_reward, actual_n, gamma, is_weights)
            mean_ntd = tf.reduce_mean(ntd_loss)
            self.update_metrics('nTD', mean_ntd)

            l2 = tf.add_n(self.online_model.losses)
            self.update_metrics('l2', l2)

            all_losses = mean_td + mean_ntd + l2
            self.update_metrics('all_losses', all_losses)

        gradients = tape.gradient(all_losses, self.online_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_variables))
        return td_loss, ntd_loss, l2, all_losses

    def td_loss(self, n_pov, q_values, n_done, n_reward, actual_n, gamma, is_weights):
        n_target = self.compute_target(n_pov, n_done, n_reward, actual_n, gamma)
        n_target = tf.expand_dims(n_target, axis=-1)
        ntd_loss = self.huber_loss(n_target, q_values, is_weights)
        return ntd_loss

    @tf.function
    def compute_target(self, next_pov, done, reward, actual_n, gamma):
        q_network = self.online_model(next_pov, training=True)
        argmax_actions = tf.argmax(q_network, axis=1, output_type='int32')
        q_target = self.target_model(next_pov, training=True)
        target = take_vector_elements(q_target, argmax_actions)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward
        return target

    def update_metrics(self, key, value):
        if key not in self.avg_metrics:
            self.avg_metrics[key] = tf.keras.metrics.Mean(name=key, dtype=tf.float32)
        self.avg_metrics[value].update_state(value)

    def target_update(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def save(self, out_dir=None):
        self.online_model.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_model.load_weights(out_dir)


class DQN(DQNbase):
    def __init__(self, action_dim, replay_buffer, build_model,
                 frames_to_update=100, update_quantity=30, update_target_net_mod=1000,
                 batch_size=32, replay_start_size=500, gamma=0.99, learning_rate=1e-4,
                 n_step=10, custom_loss=None):
        super().__init__(build_model, gamma, learning_rate, custom_loss)

        self.frames_to_update = frames_to_update
        self.update_quantity = update_quantity
        self.update_target_net_mod = update_target_net_mod
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.n_deque = deque([], maxlen=n_step)
        self.replay_buff = replay_buffer
        self.action_dim = action_dim

    def train(self, env, episodes=200, name="train/max_model.ckpt", epsilon=0.1, final_epsilon=0.01, eps_decay=0.99):
        max_reward = - np.inf
        counter = 0
        for e in range(episodes):
            start_time = timeit.default_timer()
            score, counter = self._train_episode(env, counter, epsilon)
            if len(self.replay_buff) > self.replay_start_size:
                epsilon = max(final_epsilon, epsilon * eps_decay)
            if score >= max_reward:
                max_reward = score
                self.save(name)
            stop_time = timeit.default_timer()
            print("episode: {}  score: {}  counter: {}  epsilon: {}  max: {}"
                  .format(e, score, counter, epsilon, max_reward))
            print("RunTime: ", stop_time - start_time)
            tf.summary.scalar("reward", score, step=e)
            tf.summary.flush()

    def _train_episode(self, env, current_step=0, epsilon=0.0):
        counter = current_step
        if current_step == 0:
            self.target_update()
        done, score, state = False, 0, env.reset()
        while not done:
            action, q = self.choose_act(state, epsilon, env.sample_action)
            next_state, reward, done, info = env.step(action)
            if info:
                print(info)
            score += reward
            self.perceive([q, reward, next_state, done, False])
            counter += 1
            state = next_state
            if len(self.replay_buff) > self.replay_start_size and counter % self.frames_to_update == 0:
                self.update(self.update_quantity)
            if counter % self.update_target_net_mod == 0:
                self.target_update()
        return score, counter

    def test(self, env, name="train/max_model.ckpt", number_of_trials=1, render=False, ):
        """
        Method for testing model in environment
        :param env:
        :param name:
        :param number_of_trials:
        :param render:
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
                action = self.choose_act(observation, 0, env.sample_action)
                observation, r, done, _ = env.step(action)
                if render:
                    env.render()
                reward += r
            total_reward += reward
            print("reward/avg_reward for {} trial: {}; {}".format(trial_index, reward, total_reward/(trial_index+1)))
        env.reset()
        return total_reward

    def update(self, steps, log_freq=10):
        progress = tqdm(total=steps)
        for i in range(1, steps + 1):
            progress.update(1)
            tree_idxes, minibatch, is_weights = self.replay_buff.sample(self.batch_size)

            q_values = np.array([data[0] for data in minibatch], dtype='float32')
            next_rewards = np.array([data[1] for data in minibatch], dtype='float32')
            next_pov = np.array([(np.array(data[2]) / 255) for data in minibatch], dtype='float32')
            done = np.array([data[3] for data in minibatch])
            n_pov = np.array([(np.array(data[4]) / 255) for data in minibatch], dtype='float32')
            n_reward = np.array([data[5] for data in minibatch], dtype='float32')
            n_done = np.array([data[6] for data in minibatch])
            actual_n = np.array([data[7] for data in minibatch], dtype='float32')
            gamma = np.array(self.gamma, dtype='float32')

            _, ntd_loss, _, _ = self.q_network_update(q_values, next_rewards,
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

    def perceive(self, transition):
        self.n_deque.append(transition)
        if len(self.n_deque) == self.n_deque.maxlen or transition[4]:
            while len(self.n_deque) != 0:
                n_step_pov = self.n_deque[-1][2]
                n_step_done = self.n_deque[-1][3]
                n_step_r = sum([t[1] * self.gamma ** (i + 1) for i, t in enumerate(self.n_deque)])
                self.n_deque[0].append(n_step_pov)
                self.n_deque[0].append(n_step_r)
                self.n_deque[0].append(n_step_done)
                self.n_deque[0].append(len(self.n_deque) + 1)
                self.replay_buff.store(self.n_deque.popleft())
                if not n_step_done:
                    break

    def choose_act(self, state, epsilon, action_sampler):
        inputs = (np.array(state) / 255).astype('float32')
        q_values = self.online_model(inputs[None], training=False)[0]
        if random.random() > epsilon:
            action = np.argmax(q_values).astype('int32')
        else:
            action = action_sampler()
        q = q_values[action]
        return action, q
