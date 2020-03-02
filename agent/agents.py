import os
import random

from collections import deque

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from tqdm import tqdm

from chainerrl.wrappers.atari_wrappers import LazyFrames


def classic_margin(q, action_ph, action_dim, margin, is_demo, weights):
    ae = tf.one_hot(action_ph, action_dim, on_value=0.0,
                    off_value=margin)
    max_value = tf.reduce_max(q + ae, axis=1)
    ae = tf.one_hot(action_ph, action_dim)
    j_e = tf.compat.v1.losses.absolute_difference(tf.reduce_sum(q * ae, axis=1), max_value,
                                                  weights * is_demo,
                                                  reduction='weighted_sum_over_batch_size')
    return j_e


def softmax_margin(q, action_ph, action_dim, margin, is_demo, weights):
    with tf.compat.v1.variable_scope('losses'):
        ae = tf.one_hot(action_ph, action_dim)
        j_e = tf.compat.v1.losses.softmax_cross_entropy(ae, q, weights * is_demo,
                                                        reduction='weighted_sum_over_batch_size')
    return j_e

def kl_margin(q, action_ph, action_dim, margin, is_demo, weights):
    with tf.compat.v1.variable_scope('losses'):
        ae = tf.one_hot(action_ph, action_dim)
        posterior = tf.nn.softmax(q, axis=1)
        prior = tf.nn.softmax(ae, axis=1)
        log_posterior = tf.log(posterior)
        log_prior = tf.log(prior)
        j_e = tf.reduce_sum(posterior * (log_posterior - log_prior), axis=1)
        j_e = tf.reduce_mean(j_e * weights * is_demo)
    return j_e


class DQN:
    def __init__(self, config, action_dim, replay_buffer, build_model, custom_loss=None):
        # global
        self.frame_stack = config['frame_stack']
        self.frame_skip = config['frame_skip']
        self.frames_to_update = config['frames_to_update']
        self.save_dir = config['save_dir']
        self.capacity = config['capacity']
        self.update_quantity = config['update_quantity']
        self.update_target_net_mod = config['update_target_net_mod']
        self.batch_size = config['batch_size']
        self.margin = config['margin']
        self.replay_start_size = config['replay_start_size']
        self.gamma = config['gamma']
        self.learning_rate = config['learning_rate']
        margin_mapper = {"classic_margin": classic_margin, "softmax": softmax_margin, "kl": kl_margin}
        self.j_e = margin_mapper[config['margin_func']['value']]
        self.n_deque = deque([], maxlen=config['n_step'])
        self.always_attack = config['always_attack']
        self.wandb = config['wandb']

        self.replay_buff = replay_buffer
        self.build_model = build_model
        self.custom_loss = custom_loss
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.InteractiveSession(graph=self.graph)
        self.action_dim = action_dim

        with self.sess.graph.as_default():
            self._build_graph()
            with tf.device('/cpu'):
                self.saver = tf.compat.v1.train.Saver()
            self.sess.graph.finalize()

    def _build_graph(self):
        """
        graph building and variables initialization
        :return:
        """
        self.action = tf.compat.v1.placeholder("int32", [None])
        self.reward = tf.compat.v1.placeholder("float", [None])
        self.n_step_reward = tf.compat.v1.placeholder("float", [None])
        self.done = tf.compat.v1.placeholder("float", [None])
        self.n_step_done = tf.compat.v1.placeholder("float", [None])
        self.actual_n = tf.compat.v1.placeholder("float", [None])
        self.pov = tf.compat.v1.placeholder("float", [None, 64, 64, 3 * self.frame_stack])
        self.next_pov = tf.compat.v1.placeholder("float", [None, 64, 64, 3 * self.frame_stack])
        self.n_step_pov = tf.compat.v1.placeholder("float", [None, 64, 64, 3 * self.frame_stack])
        self.is_weights = tf.compat.v1.placeholder("float", [None])

        self.q_network = self.build_model(self.pov, 'Q_network')
        self.q_target = self.build_model(self.next_pov, 'Q_target')

        q_params = tf.compat.v1.get_collection('Q_target')
        q_net_params = tf.compat.v1.get_collection('Q_net')
        self._target_update = [tf.compat.v1.assign(e, s) for e, s in zip(q_params, q_net_params)]
        self.loss, self.j_q, self.l2, self.abs_err, self.j_qn = self.all_losses()

        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def td(self, pov_ph, done_ph, reward_ph, n, with_abs=True):
        q_network = self.build_model(pov_ph, 'Q_network')
        argmax_actions = tf.argmax(q_network, axis=1)
        q_target = self.build_model(pov_ph, 'Q_target')
        with tf.compat.v1.variable_scope('{}TD'.format(n)):
            target = self.take_vector_elements(q_target, argmax_actions)
            target = self.gamma ** n * (1 - done_ph) * target + reward_ph
            q = self.take_vector_elements(self.q_network, self.action)
            j_q = tf.compat.v1.losses.huber_loss(q, target, self.is_weights, 0.4,
                                                 reduction='weighted_sum_over_batch_size')
            if with_abs:
                abs_loss = tf.abs(q - target)
                return j_q, abs_loss
            return j_q

    def all_losses(self):
        """
        collecting all losses
        :return: losses
        """
        j_q, abs_loss = self.td(self.next_pov, self.done, self.reward, 1)
        j_qn = self.td(self.n_step_pov, self.n_step_done, self.n_step_reward, self.actual_n, with_abs=False)
        l2 = tf.reduce_sum([tf.reduce_mean(reg_l) for reg_l
                            in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)])
        loss = j_q + l2 + j_qn
        if self.custom_loss:
            loss += self.custom_loss
        return loss, j_q, l2, abs_loss, j_qn

    def target_update(self):
        with self.graph.as_default():
            self.sess.run(self.target_update)

    def train(self, env, episodes=200, name="max_model.ckpt",
              epsilon=0.1, final_epsilon=0.01, eps_decay=0.99):

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
            self.log_scalar(tag="reward", value=score, step=e)
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
                self.update(self.update_quantity, self.update_quantity)
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
            print("reward/avg_reward for 2 trial: {}; {}".format(reward, total_reward/(trial_index+1)))
        env.reset()
        return total_reward

    def update(self, steps, print_steps, target_update_steps=None):
        progress = tqdm(total=steps)
        for i in range(1, steps + 1):
            progress.update(1)
            p = False
            if i % print_steps == 0:
                p = True
            self.q_network_update(print_loss=p)
            if target_update_steps and i % target_update_steps == 0:
                self.target_update()
        progress.close()

    def q_network_update(self, print_loss=False):
        """
        :param print_loss: either print loss or not
        :param proportion: proportion of samples from replay and demo buffers
        :return:
        """
        with self.graph.as_default():
            tree_idxes, minibatch, is_weights = self.replay_buff.sample(self.batch_size)

            pov_batch = [np.array(data[0]) / 256 for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_pov_batch = [np.array(data[3]) / 256 for data in minibatch]
            done_batch = [float(data[4]) for data in minibatch]
            n_pov_batch = [np.array(data[6]) / 256 for data in minibatch]
            n_reward = [data[7] for data in minibatch]
            n_done = [float(data[8]) for data in minibatch]
            actual_n = [data[9] for data in minibatch]

            feed_dict = {self.next_pov: next_pov_batch,
                         self.pov: pov_batch,
                         self.action: action_batch,
                         self.is_weights: is_weights,
                         self.reward: reward_batch,
                         self.done: done_batch,
                         self.n_step_pov: n_pov_batch,
                         self.n_step_reward: n_reward,
                         self.n_step_done: n_done,
                         self.actual_n: actual_n}

            _, abs_err, j_q, j_qn, l2 = self.sess.run(
                [self.optimizer, self.abs_err, self.j_q, self.j_qn, self.l2],
                feed_dict=feed_dict)

            if print_loss:
                print(tabulate([['TD', j_q], ['nTD', j_qn], ['L2', l2]],
                               headers=['Name', 'Value']))
            self.replay_buff.batch_update(tree_idxes, abs_err)

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
        q_values = self.compute_smth(state, self.q_network)[0]
        if random.random() <= epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_values)

    def compute_smth(self, state, smth):
        feed_dict = {self.pov: [np.array(state) / 256, ]}
        with self.graph.as_default():
            value = self.sess.run(smth, feed_dict=feed_dict)
        return value

    @staticmethod
    def take_vector_elements(vectors, indices):
        """
        For a batch of vectors, take a single vector component
        out of each vector.
        Args:
          vectors: a [batch x dims] Tensor.
          indices: an int32 Tensor with `batch` entries.
        Returns:
          A Tensor with `batch` entries, one for each vector.
        """
        return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        pass
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
        #                                              simple_value=value)])
        # self.sum_writer.add_summary(summary, step)
        # self.sum_writer.flush()

    def save(self, out_dir=None):
        with tf.device('/cpu'):
            self.saver.save(self.sess, out_dir)

    def load(self, dir_=None):
        self.saver.restore(self.sess, dir_)
