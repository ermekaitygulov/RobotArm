import numpy as np
import tensorflow as tf
from common.tf_util import take_vector_elements, huber_loss
from algorithms.base.td_base import TDPolicy


class DQN(TDPolicy):
    def __init__(self, build_model, obs_space, action_space, update_target_nn_mod=500, *args, **kwargs):

        super(DQN, self).__init__(*args, **kwargs)
        self.online_model = build_model('Online', obs_space, action_space)
        self.target_model = build_model('Target', obs_space, action_space)
        self._schedule_dict[self.target_update] = update_target_nn_mod

    def choose_act(self, state, action_sampler=None):
        inputs = {key: np.array(value)[None] for key, value in state.items()}
        q_value = self.online_model(inputs, training=False)[0]
        action = np.argmax(q_value)
        if action_sampler:
            action = action_sampler(action)
        return action, q_value[action]

    @tf.function
    def nn_update(self, state, action, next_state, done, reward,
                  n_state, n_done, n_reward, actual_n, weights,
                  gamma):
        print("Q-nn_update tracing")
        online_variables = self.online_model.trainable_variables
        with tf.GradientTape() as tape:
            q_value = self.online_model(state, training=True)
            q_value = take_vector_elements(q_value, action)
            target = self.compute_target(next_state, done, reward, 1, gamma)
            target = tf.stop_gradient(target)
            td_loss = q_value - target
            huber_td = huber_loss(td_loss)
            mean_td = tf.reduce_mean(huber_td * weights)
            self.update_metrics('TD', mean_td)

            n_target = self.compute_target(n_state, n_done, n_reward, actual_n, gamma)
            n_target = tf.stop_gradient(n_target)
            ntd_loss = q_value - n_target
            huber_ntd = huber_loss(ntd_loss)
            mean_ntd = tf.reduce_mean(huber_ntd * weights)
            self.update_metrics('nTD', mean_ntd)

            l2 = tf.add_n(self.online_model.losses)
            self.update_metrics('l2', l2)

            all_losses = mean_td + mean_ntd + l2
            self.update_metrics('all_losses', all_losses)

        gradients = tape.gradient(all_losses, online_variables)
        # for i, g in enumerate(gradients):
        #     gradients[i] = tf.clip_by_norm(g, 10)
        self.q_optimizer.apply_gradients(zip(gradients, online_variables))
        priorities = tf.abs(ntd_loss)
        return priorities

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

    def target_update(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def save(self, out_dir=None):
        self.online_model.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_model.load_weights(out_dir)
