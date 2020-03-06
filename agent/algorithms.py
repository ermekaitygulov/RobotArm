import random

from collections import deque

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.util import take_vector_elements


class DQN:
    def __init__(self, action_dim, replay_buffer, online_model, target_model,
                 frames_to_update=5, update_quantity=100, update_target_net_mod=2000,
                 batch_size=32, replay_start_size=4, gamma=0.99, learning_rate=1e-4,
                 n_step=5, custom_loss=None):
        # global
        self.frames_to_update = frames_to_update
        self.update_quantity = update_quantity
        self.update_target_net_mod = update_target_net_mod
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_deque = deque([], maxlen=n_step)

        self.replay_buff = replay_buffer
        self.online_model = online_model
        self.target_model = target_model
        self.custom_loss = custom_loss
        self.action_dim = action_dim

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.avg_metrics = dict()
        for key in ['TD', 'nTD', 'l2', 'all_losses']:
            self.avg_metrics[key] = tf.keras.metrics.Mean(name=key, dtype=tf.float32)

    def train(self, env, episodes=200, name="max_model.ckpt", epsilon=0.1, final_epsilon=0.01, eps_decay=0.99):
        max_reward = 0
        rewards_deque = deque([], maxlen=25)
        counter = 0
        for e in range(episodes):
            score, counter = self._train_episode(env, counter, epsilon)
            if len(self.replay_buff) > self.replay_start_size:
                epsilon = max(final_epsilon, epsilon * eps_decay)
            rewards_deque.append(score)
            print("episode: {}  score: {}  counter: {}  epsilon: {}  max: {}"
                  .format(e, score, counter, epsilon, max_reward))
            tf.summary.scalar("reward", score, step=e)
            if sum(filter(lambda x: x > 0, rewards_deque)):
                self.save(name)

    def _train_episode(self, env, current_step=0, epsilon=0.0):
        counter = current_step
        if current_step == 0:
            self.target_update()
        done, score, state = False, 0, env.reset()
        while done is False:
            action = self.choose_act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            score += reward
            self.perceive([state, action, reward, next_state, done, False])
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
                action = self.choose_act(observation)
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

            pov_batch = np.array([np.array(data[0]) / 255 for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch], dtype='int32')
            reward_batch = np.array([data[2] for data in minibatch], dtype='float32')
            next_pov_batch = np.array([np.array(data[3]) / 255 for data in minibatch])
            done_batch = np.array([data[4] for data in minibatch])
            n_pov_batch = np.array([np.array(data[6]) / 255 for data in minibatch])
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
            self.replay_buff.batch_update(tree_idxes, abs_loss)
        progress.close()


    @tf.function
    def q_network_update(self, pov_batch, action_batch, reward_batch,
                         next_pov_batch, done_batch, n_pov_batch,
                         n_reward, n_done, actual_n, is_weights, gamma):
        online_variables = self.online_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(online_variables)
            huber = tf.keras.losses.Huber()
            q_value = self.online_model(pov_batch, training=True)
            q_value = take_vector_elements(q_value, action_batch)
            target = self.compute_target(next_pov_batch, done_batch, reward_batch, 1, gamma)
            td_loss = huber(target, q_value, is_weights)
            print('----------------- creating metrics ---------------')
            self.avg_metrics['TD'].update_state(td_loss)

            abs_loss = tf.abs(target - q_value)

            n_target = self.compute_target(n_pov_batch, n_done, n_reward, actual_n, gamma)
            ntd_loss = huber(n_target, q_value, is_weights)
            self.avg_metrics['nTD'].update_state(ntd_loss)

            l2 = tf.add_n(self.online_model.losses)
            self.avg_metrics['l2'].update_state(l2)

            all_losses = td_loss + ntd_loss + l2
            self.avg_metrics['all_losses'].update_state(all_losses)

        gradients = tape.gradient(all_losses, online_variables)
        self.optimizer.apply_gradients(zip(gradients, online_variables))
        return abs_loss

    def compute_target(self, next_pov, done, reward, actual_n, gamma):
        q_network = self.online_model(next_pov, training=True)
        argmax_actions = tf.argmax(q_network, axis=1, output_type='int32')
        q_target = self.target_model(next_pov, training=True)
        target = take_vector_elements(q_target, argmax_actions)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward
        return target

    def target_update(self):
        self.target_model.set_weights(self.online_model.get_weights())

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
                self.replay_buff.store(self.n_deque.popleft())
                if not n_step_done:
                    break

    def choose_act(self, state, epsilon=0.01):
        inputs = np.array(state)/255
        q_values = self.online_model(inputs[None], training=False)[0]
        if random.random() <= epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_values).astype('int32')

    def save(self, out_dir=None):
        self.online_model.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_model.load_weights(out_dir)
