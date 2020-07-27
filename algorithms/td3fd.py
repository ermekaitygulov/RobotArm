from algorithms.td3 import TwinDelayedDDPG
import tensorflow as tf

from common.tf_util import update_target_variables


class TwinDelayedDDPGfD(TwinDelayedDDPG):
    @tf.function
    def nn_update(self, state, action, next_state, done, reward,
                  n_state, n_done, n_reward, actual_n, weights,
                  gamma, demo):
        priorities = self.critic_update(state, action, next_state, done, reward,
                                        n_state, n_done, n_reward, actual_n, weights,
                                        gamma)
        if tf.equal(self.q_optimizer.iterations % self.delay, 0):
            self.actor_update(state, action, demo, weights)
            update_target_variables(self.target_critic.weights, self.online_critic.weights, self.polyak)
            update_target_variables(self.target_actor.weights, self.online_actor.weights, self.polyak)
        return priorities

    @tf.function
    def actor_update(self, state, action_expert, demo, weights):
        print("Actor update tracing")
        actor_variables = self.online_actor.trainable_variables
        with tf.GradientTape() as tape:
            pre_activation = self.online_actor(state, training=True)
            action = self.scale_output(pre_activation)
            q_value, _ = self.online_critic({'state': state, 'action': action}, training=True)
            q_value_expert, _ = self.online_critic({'state': state, 'action': action_expert}, training=True)
            # Behaviour cloning
            bc_loss = self.behaviour_clone(action, action_expert, q_value, q_value_expert, demo, weights)
            self.update_metrics('bc_loss', bc_loss)
            # Actor loss
            q_value = -tf.reduce_mean(weights * q_value)
            self.update_metrics('actor_loss', q_value)
            # L2 regularization
            l2 = tf.add_n(self.online_actor.losses)
            self.update_metrics('actor_l2', l2)
            # Pre-activation regularization
            square_preactivation = tf.square(pre_activation) * self.action_reg
            square_preactivation = tf.reduce_mean(square_preactivation, axis=-1) * weights
            square_preactivation = tf.reduce_mean(square_preactivation)
            self.update_metrics('pre_activation^2', square_preactivation)
            # Final loss
            loss = q_value + l2 + square_preactivation + bc_loss
        gradients = tape.gradient(loss, actor_variables)
        for i, g in enumerate(gradients):
            self.update_metrics('Actor_Gradient_norm', tf.norm(g))
            gradients[i] = tf.clip_by_norm(g, 10)
        self.actor_optimizer.apply_gradients(zip(gradients, actor_variables))

    @staticmethod
    def behaviour_clone(action, action_expert, q_value, q_value_expert,
                        demo, weights):
        mse = tf.reduce_mean(tf.square(action - action_expert), axis=-1) * weights * demo
        mse = tf.where(q_value_expert < q_value, mse, 0.)
        mse = tf.reduce_mean(mse)
        return mse

    def perceive(self, state, action, reward, next_state, done, **kwargs):
        super(TwinDelayedDDPG, self).perceive(demo=0., state=state, action=action, reward=reward,
                                              next_state=next_state, done=done, **kwargs)

    def add_demo(self, data_loader, *args, **kwargs):
        for s, a, r, n_s, d in data_loader.sarsd_iter(*args, **kwargs):
            transition = dict(state=s, action=a, reward=r,
                              next_state=n_s, done=d)
            to_add = self.n_step(transition)
            for n_transition in to_add:
                self.replay_buff.add_demo(demo=1., **n_transition)