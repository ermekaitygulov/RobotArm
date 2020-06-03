from algorithms.ddpg.ddpg import DDPG
import tensorflow as tf
import numpy as np
from common.tf_util import update_target_variables, huber_loss
from algorithms.model import make_twin


class TD3(DDPG):
    def __init__(self, build_critic, build_actor, obs_space, action_space,
                 polyak=0.001, actor_lr=1e-4, delay=2, *args, **kwargs):
        build_twin_critic = make_twin(build_critic)
        super(TD3, self).__init__(build_twin_critic, build_actor, obs_space, action_space,
                                  polyak, actor_lr, *args, **kwargs)
        self.delay = delay

    @tf.function
    def nn_update(self, state, action, next_state, done, reward,
                  n_state, n_done, n_reward, actual_n, weights,
                  gamma):
        priorities = self.critic_update(state, action, next_state, done, reward,
                                        n_state, n_done, n_reward, actual_n, weights,
                                        gamma)
        if tf.equal(self.q_optimizer.iterations % self.delay, 0):
            self.actor_update(state, weights)
            update_target_variables(self.target_critic.weights, self.online_critic.weights, self.polyak)
            update_target_variables(self.target_actor.weights, self.online_actor.weights, self.polyak)
        return priorities

    @tf.function
    def critic_update(self, state, action, next_state, done, reward,
                      n_state, n_done, n_reward, actual_n, weights,
                      gamma):
        print("Critic update tracing")
        critic_variables = self.online_critic.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(critic_variables)
            q_value1, q_value2 = self.online_critic({'state': state, 'action': action}, training=True)
            q_value1, q_value2 = tf.squeeze(q_value1), tf.squeeze(q_value2)
            target = self.compute_target(next_state, done, reward, 1, gamma)
            target = tf.stop_gradient(target)
            td_loss1, td_loss2 = q_value1 - target, q_value2 - target
            huber_td1, huber_td2 = huber_loss(td_loss1), huber_loss(td_loss2)
            mean_td1, mean_td2 = tf.reduce_mean(huber_td1 * weights), tf.reduce_mean(huber_td2 * weights)
            self.update_metrics('TD1', mean_td1), self.update_metrics('TD2', mean_td2)

            n_target = self.compute_target(n_state, n_done, n_reward, actual_n, gamma)
            n_target = tf.stop_gradient(n_target)
            ntd_loss1, ntd_loss2 = q_value1 - n_target, q_value2 - n_target

            huber_ntd1, huber_ntd2 = huber_loss(ntd_loss1), huber_loss(ntd_loss2)
            mean_ntd1, mean_ntd2 = tf.reduce_mean(huber_ntd1 * weights), tf.reduce_mean(huber_ntd2 * weights)
            self.update_metrics('nTD1', mean_ntd1), self.update_metrics('nTD2', mean_ntd2)

            l2 = tf.add_n(self.online_critic.losses)
            self.update_metrics('critic_l2', l2)

            critic_loss = mean_td1 + mean_td2 + mean_ntd1 + mean_ntd2 + l2
            self.update_metrics('critic_loss', critic_loss)

        gradients = tape.gradient(critic_loss, critic_variables)
        for i, g in enumerate(gradients):
            self.update_metrics('Critic_Gradient_norm', tf.norm(g))
            gradients[i] = tf.clip_by_norm(g, 10)
        self.q_optimizer.apply_gradients(zip(gradients, critic_variables))
        priorities = tf.abs(ntd_loss1 + ntd_loss2)
        return priorities

    @tf.function
    def compute_target(self, next_state, done, reward, actual_n, gamma):
        print("Compute_target tracing")
        action = self.target_actor(next_state, training=True)
        target1, target2 = self.target_critic({'state': next_state, 'action': action}, training=True)
        target1, target2 = tf.squeeze(target1), tf.squeeze(target2)
        target = tf.minimum(target1, target2)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward
        return target

    @tf.function
    def actor_update(self, state, weights):
        print("Actor update tracing")
        actor_variables = self.online_actor.trainable_variables
        with tf.GradientTape() as tape:
            action = self.online_actor(state, training=True)
            q_value1, _ = self.online_critic({'state': state, 'action': action}, training=True)
            q_value = -tf.reduce_mean(weights * q_value1)
            self.update_metrics('actor_loss', q_value)
            l2 = tf.add_n(self.online_actor.losses)
            self.update_metrics('actor_l2', l2)
            loss = q_value + l2
        gradients = tape.gradient(loss, actor_variables)
        for i, g in enumerate(gradients):
            self.update_metrics('Actor_Gradient_norm', tf.norm(g))
            gradients[i] = tf.clip_by_norm(g, 10)
        self.actor_optimizer.apply_gradients(zip(gradients, actor_variables))

    def choose_act(self, state, action_sampler=None):
        inputs = {key: np.array(value)[None] for key, value in state.items()}
        action = self.online_actor(inputs, training=False)[0]
        if action_sampler:
            action = action_sampler(action)
        q_value, _ = self.online_critic({'state': inputs, 'action': action[None]}, training=False)
        q_value = q_value[0]
        return action, q_value



