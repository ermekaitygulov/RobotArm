from algorithms.ddpg import DeepDPG
import tensorflow as tf
import numpy as np
from common.tf_util import update_target_variables, huber_loss
from nn_models.building_blocks import make_twin


class TwinDelayedDDPG(DeepDPG):
    def __init__(self, build_critic, build_actor, obs_space, action_space,
                 polyak=0.001, actor_lr=1e-4, delay=2, noise_sigma=0.05, noise_clip=0.1,
                 action_reg=1., *args, **kwargs):
        self.noise_sigma = np.array(noise_sigma)
        self.noise_clip = np.array(noise_clip)
        build_twin_critic = make_twin(build_critic)
        super(TwinDelayedDDPG, self).__init__(build_twin_critic, build_actor, obs_space, action_space,
                                              polyak, actor_lr, *args, **kwargs)
        self.delay = delay
        self.action_min = action_space.low
        self.action_max = action_space.high
        self.action_reg = np.expand_dims(np.array(action_reg), axis=0)

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
        priorities = tf.abs(ntd_loss1)
        return priorities

    @tf.function
    def compute_target(self, next_state, done, reward, actual_n, gamma):
        print("Compute_target tracing")
        action = self.scale_output(self.target_actor(next_state, training=True))
        noise = tf.random.normal(action.shape) * self.noise_sigma
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        action = action + noise
        action = tf.clip_by_value(action, self.action_min, self.action_max)
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
            pre_activation = self.online_actor(state, training=True)
            action = self.scale_output(pre_activation)
            q_value1, _ = self.online_critic({'state': state, 'action': action}, training=True)
            q_value = -tf.reduce_mean(weights * q_value1)
            self.update_metrics('actor_loss', q_value)
            l2 = tf.add_n(self.online_actor.losses)
            self.update_metrics('actor_l2', l2)
            square_preactivation = tf.square(pre_activation) * self.action_reg
            square_preactivation = tf.reduce_mean(square_preactivation, axis=-1) * weights
            square_preactivation = tf.reduce_mean(square_preactivation)
            self.update_metrics('pre_activation^2', square_preactivation)
            loss = q_value + l2 + square_preactivation
        gradients = tape.gradient(loss, actor_variables)
        for i, g in enumerate(gradients):
            self.update_metrics('Actor_Gradient_norm', tf.norm(g))
            gradients[i] = tf.clip_by_norm(g, 10)
        self.actor_optimizer.apply_gradients(zip(gradients, actor_variables))

    def choose_act(self, state, action_sampler=None):
        if isinstance(state, dict):
            inputs = {key: np.array(value)[None] for key, value in state.items()}
        else:
            inputs = np.array(state)[None]
        action = self.scale_output(self.online_actor(inputs, training=False))[0]
        if action_sampler:
            action = action_sampler(action)
        q_value, _ = self.online_critic({'state': inputs, 'action': action[None]}, training=False)
        q_value = q_value[0]
        return action, q_value
