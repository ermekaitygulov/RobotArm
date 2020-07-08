import timeit
from collections import deque
import numpy as np
import tensorflow as tf


class TDPolicy:
    def __init__(self, replay_buff, train_freq=100, train_quantity=100, log_freq=100, batch_size=32,
                 replay_start_size=500, n_step=10, gamma=0.99, learning_rate=1e-4, dtype_dict=None):
        self.gamma = np.array(gamma, dtype='float32')
        self.replay_buff = replay_buff
        self.train_freq = train_freq
        self.train_quantity = train_quantity
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.n_deque = deque([], maxlen=n_step)
        self.avg_metrics = dict()

        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.priorities_store = list()
        if dtype_dict is not None:
            ds = tf.data.Dataset.from_generator(self.sample_generator, output_types=dtype_dict)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            self.sampler = ds.take
        else:
            self.sampler = self.sample_generator
        self.run_time_deque = deque(maxlen=log_freq)
        self._schedule_dict = dict()
        self._schedule_dict[self.update_log] = log_freq
        self.online_models = list()
        self.target_models = list()

    # RL methods
    def train(self, env, episodes=200, save_dir="train/max_model/", save_window=25):
        max_reward = - np.inf
        counter = 0
        window = deque([], maxlen=save_window)
        for e in range(episodes):
            start_time = timeit.default_timer()
            score, counter = self._train_episode(env, counter)
            window.append(score)
            avg_reward = sum(window) / len(window)
            if avg_reward >= max_reward:
                print("MaxAvg reward moved from {:.2f} to {:.2f} (save model)".format(max_reward, avg_reward))
                max_reward = avg_reward
                self.save(save_dir)
            stop_time = timeit.default_timer()
            print("episode: {}  score: {}  counter: {}  max: {}"
                  .format(e, score, counter, max_reward))
            print("RunTime: ", stop_time - start_time)
            tf.summary.scalar("reward", score, step=e)
            tf.summary.flush()

    def _train_episode(self, env, current_step=0):
        counter = current_step
        done, score, state = False, 0, env.reset()
        while not done:
            action, _ = self.choose_act(state, env.sample_action)
            next_state, reward, done, info = env.step(action)
            score += reward
            self.perceive(state, action, reward, next_state, done)
            counter += 1
            state = next_state
            if self.replay_buff.get_stored_size() > self.replay_start_size and counter % self.train_freq == 0:
                self.update(self.train_quantity)
        return score, counter

    def perceive(self, state, action, reward, next_state, done, **kwargs):
        transition = dict(state=state, action=action, reward=reward,
                          next_state=next_state, done=done, **kwargs)
        to_add = self.n_step(transition)
        for n_transition in to_add:
            self.replay_buff.add(**n_transition)

    def n_step(self, transition):
        self.n_deque.append(transition)
        to_add = list()
        if len(self.n_deque) == self.n_deque.maxlen or transition['done']:
            while len(self.n_deque) != 0:
                n_step_state = self.n_deque[-1]['next_state']
                n_step_done = self.n_deque[-1]['done']
                n_step_r = sum([t['reward'] * self.gamma ** i for i, t in enumerate(self.n_deque)])
                self.n_deque[0]['n_state'] = n_step_state
                self.n_deque[0]['n_reward'] = n_step_r
                self.n_deque[0]['n_done'] = n_step_done
                self.n_deque[0]['actual_n'] = len(self.n_deque)
                to_add.append(self.n_deque.popleft())
                if not n_step_done:
                    break
        return to_add

    def choose_act(self, state, action_sampler=None):
        raise NotImplementedError

    # Update methods
    def update(self, steps):
        start_time = timeit.default_timer()
        for batch in self.sampler(steps):
            indexes = batch.pop('indexes')
            priorities = self.nn_update(gamma=self.gamma, **batch)
            self.priorities_store.append({'indexes': indexes.numpy(),
                                          'priorities': priorities.numpy()})
            self.schedule()
            stop_time = timeit.default_timer()
            self.run_time_deque.append(stop_time - start_time)
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

    def nn_update(self, state, action, next_state, done, reward,
                  n_state, n_done, n_reward, actual_n, weights,
                  gamma):
        raise NotImplementedError

    def compute_target(self, next_state, done, reward, actual_n, gamma):
        raise NotImplementedError

    # Log methods
    def update_metrics(self, key, value):
        if key not in self.avg_metrics:
            self.avg_metrics[key] = tf.keras.metrics.Mean(name=key, dtype=tf.float32)
        self.avg_metrics[key].update_state(value)

    def update_log(self):
        update_frequency = len(self.run_time_deque) / sum(self.run_time_deque)
        print("LearnerEpoch({:.2f}it/sec): ".format(update_frequency), self.q_optimizer.iterations.numpy())
        for key, metric in self.avg_metrics.items():
            tf.summary.scalar(key, metric.result(), step=self.q_optimizer.iterations)
            print('  {}:     {:.5f}'.format(key, metric.result()))
            metric.reset_states()
        tf.summary.flush()

    def schedule(self):
        for key, value in self._schedule_dict.items():
            if tf.equal(self.q_optimizer.iterations % value, 0):
                key()

    # Save Load
    def save(self, out_dir=None):
        raise NotImplementedError

    def load(self, path=None):
        raise NotImplementedError

    # Functions below are needed for apex
    def get_online(self):
        return [model.get_weights() for model in self.online_models]

    def get_target(self):
        return [model.get_weights() for model in self.target_models]

    def set_weights(self, online_weights, target_weights):
        for model, weights in zip(self.online_models, online_weights):
            model.set_weights(weights)
        for model, weights in zip(self.target_models, target_weights):
            model.set_weights(weights)
