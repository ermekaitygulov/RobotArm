import timeit

from algorithms.dqn import DQN
from algorithms.model import ClassicCnn, DuelingModel, MLP
from numpy import random
import numpy as np
import os
import tensorflow as tf

from utils.util import take_vector_elements, huber_loss


class Dataset:
    def __init__(self, steps, batch_size):
        self.data = dict()
        self.data['state'] = {'pov': random.randint(0, 255, size=(31*batch_size, 256, 256, 12)).astype('uint8'),
                              'angles': random.uniform(-2*np.pi, -2*np.pi, size=(31*batch_size, 6))}
        self.data['action'] = np.ones(31*batch_size, dtype='int32')
        self.data['reward'] = np.ones(31*batch_size, dtype='float32')
        self.data['done'] = np.ones(31*batch_size, dtype='bool')
        self.data['n_reward'] = np.ones(31*batch_size, dtype='float32')
        self.data['n_done'] = np.ones(31*batch_size, dtype='bool')
        self.data['actual_n'] = 5*np.ones(31*batch_size, dtype='float32')
        self.data['weights'] = np.ones(31*batch_size, dtype='float32')
        self.step = 0

    def sample(self, batch_size):
        minibatch = {key: value[:+batch_size] for key, value in self.data.items()
                     if 'state' not in key}
        minibatch['state'] = {key: value[:batch_size] for key, value in self.data['state'].items()}
        minibatch['next_state'] = {key: value[:batch_size] for key, value in
                              self.data['state'].items()}
        minibatch['n_state'] = {key: value[:batch_size] for key, value in
                                   self.data['state'].items()}
        return minibatch


class TestAgent(DQN):
    def update(self, steps):
        start_time = timeit.default_timer()
        ds = self.replay_buff.sample(self.batch_size * steps)
        ds['state'] = self.preprocess_state(ds['state'])
        ds['next_state'] = self.preprocess_state(ds['next_state'])
        ds['n_state'] = self.preprocess_state(ds['n_state'])
        loss_list = list()
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.batch(self.batch_size)
        # ds = ds.map(self.preprocess_ds)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        for batch in ds:
            _, ntd_loss, _, _ = self.q_network_update(gamma=self.gamma, **batch)
            stop_time = timeit.default_timer()
            self._run_time_deque.append(stop_time - start_time)
            self.schedule()
            loss_list.append(np.abs(ntd_loss.numpy()))
            start_time = timeit.default_timer()





def make_model(name, obs_space, action_space):
    pov = tf.keras.Input(shape=(256, 256, 12))
    angles = tf.keras.Input(shape=6)
    pov_base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])(pov)
    angles_base = MLP([512, 256])(angles)
    base = tf.keras.layers.concatenate([pov_base, angles_base])
    head = DuelingModel([1024], 6)(base)
    model = tf.keras.Model(inputs={'pov': pov, 'angles': angles}, outputs=head, name=name)
    return model


def profiling_simple_dqn(update_number=100, batch_size=32):
    tf.config.optimizer.set_jit(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    dataset = Dataset(update_number, batch_size=batch_size)
    agent = TestAgent(dataset, make_model, (256, 256, 12), 6, log_freq=10, batch_size=batch_size)
    print("Starting Profiling")
    with tf.profiler.experimental.Profile('train/'):
        for i in range(update_number//30):
            agent.update(30)
    while True:
        continue


def profiling_data_dqn(update_number=15, batch_size=32):
    tf.debugging.set_log_device_placement(False)
    tf.config.optimizer.set_jit(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    agent = TestAgent(None, make_model, (256, 256, 12), 6, log_freq=10, batch_size=batch_size)
    dataset = Dataset(update_number, batch_size=batch_size)
    ds = tf.data.Dataset.from_tensor_slices(dataset.data)
    dtype_dict = agent.dtype_dict

    def preprocess_ds(sample):
        casted_sample = dict()
        for key, value in sample.items():
            casted_sample[key] = tf.cast(value, dtype=dtype_dict[key])
        return casted_sample
    ds = ds.map(preprocess_ds)
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    print("Starting Profiling")
    with tf.profiler.experimental.Profile('train/'):
        start_time = timeit.default_timer()
        for batch in ds:
            agent.batch_update(batch)
            stop_time = timeit.default_timer()
            agent._run_time_deque.append(1/(stop_time - start_time))
            start_time = timeit.default_timer()
    while True:
        continue


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    profiling_simple_dqn(100, 32)
