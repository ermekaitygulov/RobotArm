from collections import deque

import numpy as np
import tensorflow as tf
import timeit

from common.tf_util import huber_loss, update_target_variables


class DDPG:
    def __init__(self, replay_buffer, build_critic, build_actor, obs_space, action_space, dtype_dict,
                 train_freq=100, train_quantity=100, log_freq=50, polyak=0.99, batch_size=32,
                 replay_start_size=500, gamma=0.99, learning_rate=1e-4, n_step=10):

        self.gamma = np.array(gamma, dtype='float32')
        self.online_critic = build_critic('Online_Q', obs_space, action_space)
        self.target_critic = build_critic('Target_Q', obs_space, action_space)
        update_target_variables(self.target_critic.weights, self.online_critic.weights)
        self.online_actor = build_actor('Online_actor', obs_space, action_space)
        self.target_actor = build_actor('Target_actor', obs_space, action_space)
        update_target_variables(self.target_actor.weights, self.online_actor.weights)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.avg_metrics = dict()
        self.train_freq = train_freq
        self.train_quantity = train_quantity
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.n_deque = deque([], maxlen=n_step)
        self.replay_buff = replay_buffer
        self.polyak = polyak

        self.priorities_store = list()
        if dtype_dict is not None:
            ds = tf.data.Dataset.from_generator(self.sample_generator, output_types=dtype_dict)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            self.sampler = ds.take
        else:
            self.sampler = self.sample_generator

        self._run_time_deque = deque(maxlen=log_freq)
        self._schedule_dict = dict()
        self._schedule_dict[self.update_log] = log_freq
        # TODO tune OUNoise
        self._exploration = OUNoise(action_space.shape[0], action_space.low, action_space.high)

    def train(self, env, episodes=200, name="train/max_model.ckpt", save_window=20):
        max_reward = - np.inf
        counter = 0
        window = deque([], maxlen=save_window)
        for e in range(episodes):
            start_time = timeit.default_timer()
            score, counter = self._train_episode(env, counter)
            window.append(score)
            avg_reward = sum(window)/len(window)
            if avg_reward >= max_reward:
                print("MaxAvg reward moved from {:.2f} to {:.2f} (save model)".format(max_reward, avg_reward))
                max_reward = avg_reward
                self.save(name)
            stop_time = timeit.default_timer()
            print("episode: {}  score: {}  counter: {}  maxavg: {}"
                  .format(e, score, counter, max_reward))
            print("RunTime: ", stop_time - start_time)
            tf.summary.scalar("reward", score, step=e)
            tf.summary.flush()

    def _train_episode(self, env, current_step=0):
        counter = current_step
        done, score, state = False, 0, env.reset()
        while not done:
            action = self.choose_act(state)
            if self._exploration:
                action = self._exploration.get_action(action, counter)
            next_state, reward, done, info = env.step(action)
            if info:
                print(info)
            score += reward
            self.perceive(state, action, reward, next_state, done)
            counter += 1
            state = next_state
            if self.replay_buff.get_stored_size() > self.replay_start_size \
                    and counter % self.train_freq == 0:
                self.update(self.train_quantity)
        return score, counter

    def update(self, steps):
        start_time = timeit.default_timer()
        for batch in self.sampler(steps):
            indexes = batch.pop('indexes')
            priorities = self.nn_update(gamma=self.gamma, **batch)
            self.priorities_store.append({'indexes': indexes.numpy(),
                                          'priorities': priorities.numpy()})
            self.schedule()
            stop_time = timeit.default_timer()
            self._run_time_deque.append(stop_time - start_time)
            start_time = timeit.default_timer()
        while len(self.priorities_store) > 0:
            priorities = self.priorities_store.pop(0)
            self.replay_buff.update_priorities(**priorities)

    def sample_generator(self, steps=None):
        steps_done = 0
        finite_loop = bool(steps)
        steps = steps if finite_loop else 1
        while steps_done < steps:
            yield self.replay_buff.sample(self.batch_size)
            if len(self.priorities_store) > 0:
                priorities = self.priorities_store.pop(0)
                self.replay_buff.update_priorities(**priorities)
            steps += int(finite_loop)

    def choose_act(self, state):
        inputs = {key: np.array(value)[None] for key, value in state.items()}
        action = self.online_actor(inputs, training=False)[0]
        return action

    @tf.function
    def nn_update(self, state, action, next_state, done, reward,
                  n_state, n_done, n_reward, actual_n, weights,
                  gamma):
        priorities = self.critic_update(state, action, next_state, done, reward,
                                        n_state, n_done, n_reward, actual_n, weights,
                                        gamma)
        self.actor_update(state)
        update_target_variables(self.target_critic.weights, self.online_critic.weights, self.polyak)
        update_target_variables(self.target_actor.weights, self.online_actor.weights, self.polyak)
        return priorities

    @tf.function
    def actor_update(self, state):
        print("Actor update tracing")
        actor_variables = self.online_actor.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(actor_variables)
            action = self.online_actor(state, training=True)
            q_value = self.online_critic({'state': state, 'action': action}, training=True)
            self.update_metrics('actor_loss', q_value)
            l2 = tf.add_n(self.online_actor.losses)
            self.update_metrics('actor_l2', l2)
            loss = -q_value + l2
        gradients = tape.gradient(loss, actor_variables)
        for i, g in enumerate(gradients):
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
            target = self.compute_target(next_state, done, reward, 1, gamma)
            td_loss = q_value - target
            huber_td = huber_loss(td_loss)
            mean_td = tf.reduce_mean(huber_td * weights)
            self.update_metrics('TD', mean_td)

            n_target = self.compute_target(n_state, n_done, n_reward, actual_n, gamma)
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
            gradients[i] = tf.clip_by_norm(g, 10)
        self.critic_optimizer.apply_gradients(zip(gradients, critic_variables))
        priorities = tf.abs(ntd_loss)
        return priorities

    @tf.function
    def compute_target(self, next_state, done, reward, actual_n, gamma):
        print("Compute_target tracing")
        action = self.target_actor(next_state, training=True)
        target = self.target_critic({'state': next_state, 'action': action}, training=True)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward
        return target

    def update_metrics(self, key, value):
        if key not in self.avg_metrics:
            self.avg_metrics[key] = tf.keras.metrics.Mean(name=key, dtype=tf.float32)
        self.avg_metrics[key].update_state(value)

    def perceive(self, state, action, reward, next_state, done, **kwargs):
        transition = dict(state=state, action=action, reward=reward,
                          next_state=next_state, done=done, **kwargs)
        self.n_deque.append(transition)
        if len(self.n_deque) == self.n_deque.maxlen or transition['done']:
            while len(self.n_deque) != 0:
                n_step_state = self.n_deque[-1]['next_state']
                n_step_done = self.n_deque[-1]['done']
                n_step_r = sum([t['reward'] * self.gamma ** i for i, t in enumerate(self.n_deque)])
                self.n_deque[0]['n_state'] = n_step_state
                self.n_deque[0]['n_reward'] = n_step_r
                self.n_deque[0]['n_done'] = n_step_done
                self.n_deque[0]['actual_n'] = len(self.n_deque)
                self.replay_buff.add(**self.n_deque.popleft())
                if not n_step_done:
                    break

    def schedule(self):
        for key, value in self._schedule_dict.items():
            if tf.equal(self.critic_optimizer.iterations % value, 0):
                key()

    def update_log(self):
        update_frequency = len(self._run_time_deque) / sum(self._run_time_deque)
        print("LearnerEpoch({:.2f}it/sec): ".format(update_frequency), self.critic_optimizer.iterations.numpy())
        for key, metric in self.avg_metrics.items():
            tf.summary.scalar(key, metric.result(), step=self.critic_optimizer.iterations)
            print('  {}:     {:.5f}'.format(key, metric.result()))
            metric.reset_states()
        tf.summary.flush()

    def save(self, out_dir=None):
        self.online_critic.save_weights(out_dir)
        self.online_actor.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_critic.load_weights(out_dir)
        self.online_actor.load_weights(out_dir)


class OUNoise(object):
    def __init__(self, action_dim, low, high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.1, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.low = low
        self.high = high
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

