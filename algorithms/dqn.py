import random
from collections import deque
from copy import deepcopy

import numpy as np
import tensorflow as tf
import timeit

from utils.util import take_vector_elements, huber_loss


class DQN:
    dtype_dict = {'state': {'pov': 'float32', 'angles': 'float32'},
                  'action': 'int32',
                  'reward': 'float32',
                  'next_state': {'pov': 'float32', 'angles': 'float32'},
                  'done': 'bool',
                  'n_state': {'pov': 'float32', 'angles': 'float32'},
                  'n_reward': 'float32',
                  'n_done': 'bool',
                  'actual_n': 'float32',
                  'weights': 'float32'}

    def __init__(self, replay_buffer, build_model, obs_shape, action_shape, train_freq=100, train_quantity=100,
                 log_freq=50, update_target_nn_mod=500, batch_size=32, replay_start_size=500, gamma=0.99,
                 learning_rate=1e-4, n_step=10, custom_loss=None):

        self.gamma = np.array(gamma, dtype='float32')
        self.online_model = build_model('Online', obs_shape, action_shape)
        self.target_model = build_model('Target', obs_shape, action_shape)
        self.custom_loss = custom_loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.huber_loss = tf.keras.losses.Huber(1.0, tf.keras.losses.Reduction.NONE)
        self.avg_metrics = dict()
        self.train_freq = train_freq
        self.train_quantity = train_quantity
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.n_deque = deque([], maxlen=n_step)
        self.replay_buff = replay_buffer

        self._update_frequency = 0
        self._run_time_deque = deque(maxlen=log_freq)
        self._schedule_dict = dict()
        self._schedule_dict[self.target_update] = update_target_nn_mod
        self._schedule_dict[self.update_log] = log_freq

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
            action, _ = self.choose_act(state, epsilon, env.sample_action)
            next_state, reward, done, info = env.step(action)
            if info:
                print(info)
            score += reward
            self.perceive(state, action, reward, next_state, done)
            counter += 1
            state = next_state
            if len(self.replay_buff) > self.replay_start_size and counter % self.train_freq == 0:
                self.update(self.train_quantity)
        return score, counter

    def test(self, env, name="train/max_model.ckpt", number_of_trials=1, logging=False):
        """
        Method for testing model in environment
        :param env:
        :param name:
        :param number_of_trials:
        :param logging:
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
                action, _ = self.choose_act(observation, 0, env.sample_action)
                observation, r, done, _ = env.step(action)
                reward += r
            total_reward += reward
            if logging:
                print("reward/avg_reward for {} trial: {}; {}".
                      format(trial_index, reward, total_reward/(trial_index+1)))
        env.reset()
        return total_reward

    def update(self, steps):
        start_time = timeit.default_timer()
        ds = self.replay_buff.sample(self.batch_size*steps)
        indexes = ds.pop('indexes')
        loss_list = list()
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.map(self.preprocess_ds)
        ds = ds.batch(self.batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        for batch in ds:
            _, ntd_loss, _, _ = self.q_network_update(gamma=self.gamma, **batch)
            stop_time = timeit.default_timer()
            self._run_time_deque.append(stop_time - start_time)
            self.schedule()
            loss_list.append(np.abs(ntd_loss.numpy()))
            start_time = timeit.default_timer()
        self.replay_buff.update_priorities(indexes, np.concatenate(loss_list))

    def preprocess_ds(self, sample):
        sample['state'] = self.preprocess_state(sample['state'])
        sample['next_state'] = self.preprocess_state(sample['next_state'])
        sample['n_state'] = self.preprocess_state(sample['n_state'])
        return sample

    def dict_cast(self, dictionary, dtype_dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                dictionary[key] = self.dict_cast(value, dtype_dict[key])
            else:
                dictionary[key] = tf.cast(value, dtype=dtype_dict[key])
            if 'state' in key:
                dictionary[key] = self.preprocess_state(dictionary[key])
        return dictionary

    def choose_act(self, state, epsilon, action_sampler):
        inputs = {key: np.array(value).astype('float32')[None] for key, value in state.items()}
        inputs = self.preprocess_state(inputs)
        q_value = self.online_model(inputs, training=False)[0]
        if random.random() > epsilon:
            action = np.argmax(q_value).astype('int32')
        else:
            action = action_sampler()
        return action, q_value[action]

    @tf.function
    def q_network_update(self, state, action, reward, next_state, done, n_state,
                         n_reward, n_done, actual_n, weights, gamma):
        print("Q-nn_update tracing")
        online_variables = self.online_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(online_variables)
            q_values = self.online_model(state, training=True)
            q_values = take_vector_elements(q_values, action)
            td_loss = self.td_loss(next_state, q_values, done, reward, 1, gamma)
            huber_td = huber_loss(td_loss)
            mean_td = tf.reduce_mean(huber_td * weights)
            self.update_metrics('TD', mean_td)

            ntd_loss = self.td_loss(n_state, q_values, n_done, n_reward, actual_n, gamma)
            huber_ntd = huber_loss(ntd_loss)
            mean_ntd = tf.reduce_mean(huber_ntd * weights)
            self.update_metrics('nTD', mean_ntd)

            l2 = tf.add_n(self.online_model.losses)
            self.update_metrics('l2', l2)

            all_losses = mean_td + mean_ntd + l2
            self.update_metrics('all_losses', all_losses)

        gradients = tape.gradient(all_losses, online_variables)
        for i, g in enumerate(gradients):
            gradients[i] = tf.clip_by_norm(g, 10)
        self.optimizer.apply_gradients(zip(gradients, online_variables))
        return td_loss, ntd_loss, l2, all_losses

    @tf.function
    def td_loss(self, n_state, q_values, n_done, n_reward, actual_n, gamma):
        print("TD-Loss tracing")
        n_target = self.compute_target(n_state, n_done, n_reward, actual_n, gamma)
        n_target = tf.stop_gradient(n_target)
        ntd_loss = q_values - n_target
        return ntd_loss

    @tf.function
    def compute_target(self, next_state, done, reward, actual_n, gamma):
        print("Compute_target tracing")
        q_network = self.online_model(next_state, training=True)
        argmax_actions = tf.argmax(q_network, axis=1, output_type='int32')
        q_target = self.target_model(next_state, training=True)
        target = take_vector_elements(q_target, argmax_actions)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward
        return target

    def update_metrics(self, key, value):
        if key not in self.avg_metrics:
            self.avg_metrics[key] = tf.keras.metrics.Mean(name=key, dtype=tf.float32)
        self.avg_metrics[key].update_state(value)

    def target_update(self):
        # TODO remove print
        print("target update")
        self.target_model.set_weights(self.online_model.get_weights())

    def perceive(self, state, action, reward, next_state, done, **kwargs):
        transition = dict(state=state, action=action, reward=reward,
                          next_state=next_state, done=done, **kwargs)
        self.n_deque.append(transition)
        if len(self.n_deque) == self.n_deque.maxlen or transition['done']:
            while len(self.n_deque) != 0:
                n_step_state = self.n_deque[-1]['next_state']
                n_step_done = self.n_deque[-1]['done']
                n_step_r = sum([t['reward'] * self.gamma ** (i + 1) for i, t in enumerate(self.n_deque)])
                self.n_deque[0]['n_state'] = n_step_state
                self.n_deque[0]['n_reward'] = n_step_r
                self.n_deque[0]['n_done'] = n_step_done
                self.n_deque[0]['actual_n'] = len(self.n_deque) + 1
                self.replay_buff.add(self.n_deque.popleft())
                if not n_step_done:
                    break

    def schedule(self):
        return_dict = {key: None for key in self._schedule_dict.keys()}
        for key, value in self._schedule_dict.items():
            if tf.equal(self.optimizer.iterations % value, 0):
                return_dict[key] = key()
        return return_dict

    def preprocess_state(self, state):
        state['pov'] /= 255
        return state

    def update_log(self):
        update_frequency = len(self._run_time_deque) / sum(self._run_time_deque)
        print("LearnerEpoch({:.2f}it/sec): ".format(update_frequency), self.optimizer.iterations.numpy())
        for key, metric in self.avg_metrics.items():
            tf.summary.scalar(key, metric.result(), step=self.optimizer.iterations)
            print('  {}:     {:.5f}'.format(key, metric.result()))
            metric.reset_states()
        tf.summary.flush()

    def save(self, out_dir=None):
        self.online_model.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_model.load_weights(out_dir)
