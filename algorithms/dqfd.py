from algorithms.dqn import DoubleDuelingDQN
import tensorflow as tf
from common.tf_util import take_vector_elements, huber_loss


class DQfromDemonstrations(DoubleDuelingDQN):
    def __init__(self, margin, *args, **kwargs):
        super(DQfromDemonstrations, self).__init__(*args, **kwargs)
        self.margin_value = margin

    @tf.function
    def nn_update(self, state, action, next_state, done, reward,
                  n_state, n_done, n_reward, actual_n, weights,
                  gamma, demo):
        print("Q-nn_update tracing")
        online_variables = self.online_model.trainable_variables
        with tf.GradientTape() as tape:
            q_values = self.online_model(state, training=True)

            margin = self.margin_loss(q_values, action, demo, weights)
            self.update_metrics('Margin', margin)

            q_value = take_vector_elements(q_values, action)
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

            all_losses = mean_td + mean_ntd + margin + l2
            self.update_metrics('all_losses', all_losses)

        gradients = tape.gradient(all_losses, online_variables)

        for i, g in enumerate(gradients):
            self.update_metrics('Gradient_norm', tf.norm(g))
            gradients[i] = tf.clip_by_norm(g, 10)

        self.q_optimizer.apply_gradients(zip(gradients, online_variables))
        priorities = tf.abs(ntd_loss)
        return priorities

    def margin_loss(self, q_values, action, demo, weights):
        margin = tf.one_hot(action, q_values.shape[1], on_value=0.0,
                            off_value=self.margin_value)
        margin = tf.cast(margin, 'float32')
        max_value = tf.reduce_max(q_values + margin, axis=1)
        expert_value = take_vector_elements(q_values, action)
        j_e = tf.abs(expert_value - max_value)
        j_e = tf.reduce_mean(j_e * weights * demo)
        return j_e

    def perceive(self, **kwargs):
        super(DQfromDemonstrations, self).perceive(demo=0., **kwargs)

    def add_demo(self, data_loader, *args, **kwargs):
        for s, a, r, n_s, d in data_loader.sarsd_iter(*args, **kwargs):
            transition = dict(state=s, action=a, reward=r,
                              next_state=n_s, done=d)
            to_add = self.n_step(transition)
            for n_transition in to_add:
                self.replay_buff.add_demo(demo=1., **n_transition)
