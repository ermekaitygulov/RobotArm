import os

import numpy as np
import tensorflow as tf
from algorithms.td_base import TDPolicy
from common.tf_util import huber_loss, update_target_variables


class DeepDPG(TDPolicy):
    def __init__(self, build_critic, build_actor, obs_space, action_space,
                 polyak=0.005, actor_lr=1e-4, *args, **kwargs):
        super(DeepDPG, self).__init__(*args, **kwargs)
        self.online_critic = build_critic('Online_Q', obs_space, action_space)
        self.online_models.append(self.online_critic)
        self.target_critic = build_critic('Target_Q', obs_space, action_space)
        self.target_models.append(self.target_critic)
        update_target_variables(self.target_critic.weights, self.online_critic.weights)
        self.online_actor = build_actor('Online_actor', obs_space, action_space)
        self.online_models.append(self.online_actor)
        self.target_actor = build_actor('Target_actor', obs_space, action_space)
        self.target_models.append(self.target_actor)
        update_target_variables(self.target_actor.weights, self.online_actor.weights)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.polyak = polyak
        self.action_space = action_space

    @tf.function
    def scale_output(self, x):
        x = tf.keras.activations.tanh(x)
        x = x * (self.action_space.high - self.action_space.low) / 2\
            + (self.action_space.high + self.action_space.low) / 2
        return x

    def choose_act(self, state, action_sampler=None):
        if isinstance(state, dict):
            inputs = {key: np.array(value)[None] for key, value in state.items()}
        else:
            inputs = np.array(state)[None]
        action = self.scale_output(self.online_actor(inputs, training=False))[0]
        if action_sampler:
            action = action_sampler(action)
        q_value = self.online_critic({'state': inputs, 'action': action[None]}, training=False)[0]
        return action, q_value

    @tf.function
    def nn_update(self, state, action, next_state, done, reward,
                  n_state, n_done, n_reward, actual_n, weights,
                  gamma):
        priorities = self.critic_update(state, action, next_state, done, reward,
                                        n_state, n_done, n_reward, actual_n, weights,
                                        gamma)
        self.actor_update(state, weights)
        update_target_variables(self.target_critic.weights, self.online_critic.weights, self.polyak)
        update_target_variables(self.target_actor.weights, self.online_actor.weights, self.polyak)
        return priorities

    @tf.function
    def actor_update(self, state, weights):
        print("Actor update tracing")
        actor_variables = self.online_actor.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(actor_variables)
            action = self.scale_output(self.online_actor(state, training=True))
            q_value = -tf.reduce_mean(weights * self.online_critic({'state': state, 'action': action}, training=True))
            self.update_metrics('actor_loss', q_value)
            l2 = tf.add_n(self.online_actor.losses)
            self.update_metrics('actor_l2', l2)
            loss = q_value + l2
        gradients = tape.gradient(loss, actor_variables)
        for i, g in enumerate(gradients):
            self.update_metrics('Actor_Gradient_norm', tf.norm(g))
            gradients[i] = tf.clip_by_norm(g, 10)
        self.actor_optimizer.apply_gradients(zip(gradients, actor_variables))

    @tf.function
    def critic_update(self, state, action, next_state, done, reward,
                      n_state, n_done, n_reward, actual_n, weights,
                      gamma):
        print("Critic update tracing")
        critic_variables = self.online_critic.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(critic_variables)
            q_value = self.online_critic({'state': state, 'action': action}, training=True)
            q_value = tf.squeeze(q_value)
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

            l2 = tf.add_n(self.online_critic.losses)
            self.update_metrics('critic_l2', l2)

            critic_loss = mean_td + mean_ntd + l2
            self.update_metrics('critic_loss', critic_loss)

        gradients = tape.gradient(critic_loss, critic_variables)
        for i, g in enumerate(gradients):
            self.update_metrics('Critic_Gradient_norm', tf.norm(g))
            gradients[i] = tf.clip_by_norm(g, 10)
        self.q_optimizer.apply_gradients(zip(gradients, critic_variables))
        priorities = tf.abs(ntd_loss)
        return priorities

    @tf.function
    def compute_target(self, next_state, done, reward, actual_n, gamma):
        print("Compute_target tracing")
        action = self.scale_output(self.target_actor(next_state, training=True))
        target = self.target_critic({'state': next_state, 'action': action}, training=True)
        target = tf.squeeze(target)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward
        return target

    def save(self, out_dir=None):
        name = self.online_critic.name + '.ckpt'
        self.online_critic.save_weights(os.path.join(out_dir, name))
        name = self.online_actor.name + '.ckpt'
        self.online_actor.save_weights(os.path.join(out_dir, name))

    def load(self, online_actor_path=None, online_critic_path=None,
             target_actor_path=None, target_critic_path=None):
        if online_actor_path:
            self.online_actor.load_weights(online_actor_path)
        if online_critic_path:
            self.online_critic.load_weights(online_critic_path)
        if target_actor_path:
            self.target_actor.load_weights(target_actor_path)
        if target_critic_path:
            self.target_critic.load_weights(target_critic_path)
