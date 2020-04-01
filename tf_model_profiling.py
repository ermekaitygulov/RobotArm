import timeit

from algorithms.dqn import DQN
from algorithms.model import ClassicCnn, DuelingModel
from numpy import random
import os
import tensorflow as tf


class Dataset:
    def __init__(self, steps, batch_size):
        self.data = dict()
        self.data['state'] = random.randint(0, 255, size=(steps*batch_size, 256, 256, 12), dtype='uint8')
        self.data['action'] = random.randint(0, 5, size=(steps*batch_size))
        self.data['reward'] = random.randint(0, 10, size=(steps*batch_size))
        self.data['next_state'] = random.randint(0, 255, size=(steps*batch_size, 256, 256, 12), dtype='uint8')
        self.data['done'] = random.randint(0, 1, size=(steps*batch_size))
        self.data['n_state'] = random.randint(0, 255, size=(steps*batch_size, 256, 256, 12), dtype='uint8')
        self.data['n_reward'] = random.randint(0, 10, size=(steps*batch_size))
        self.data['n_done'] = random.randint(0, 1, size=(steps*batch_size))
        self.data['actual_n'] = random.randint(0, 5, size=(steps*batch_size))
        self.data['weights'] = random.uniform(size=[steps*batch_size])
        self.step = 0

    def sample(self, batch_size):
        minibatch = {key: value[self.step:(self.step+batch_size)] for key, value in self.data.items()}
        idx = None
        return idx, minibatch

    def update_priorities(self, *args, **kwargs):
        pass


class TestAgent(DQN):
    def batch_update(self, batch):
        start_time = timeit.default_timer()
        _, ntd_loss, _, _ = self.q_network_update(batch['state'], batch['action'],
                                                  batch['reward'], batch['next_state'],
                                                  batch['done'], batch['n_state'],
                                                  batch['n_reward'], batch['n_done'],
                                                  batch['actual_n'], batch['weights'], self.gamma)

        stop_time = timeit.default_timer()
        self._run_time_deque.append(stop_time - start_time)
        self.schedule()
        return ntd_loss


def make_model(name, input_shape, output_shape):
    base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])
    head = DuelingModel([1024], output_shape)
    model = tf.keras.Sequential([base, head], name)
    model.build((None, ) + input_shape)
    return model


def profiling_simple_dqn(update_number=100, batch_size=32):
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

    dataset = Dataset(update_number, batch_size=batch_size)

    agent = TestAgent(None, make_model, (256, 256, 12), 6, log_freq=10, batch_size=batch_size)
    print("Starting Profiling")
    with tf.profiler.experimental.Profile('train/'):
        for i in range(update_number):
            agent.batch_update(dataset.sample(batch_size))
    while True:
        continue


def profiling_data_dqn(update_number=100, batch_size=32):
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

    dataset = Dataset(update_number, batch_size=batch_size)
    ds = tf.data.Dataset.from_tensor_slices(dataset.data.values())
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    agent = TestAgent(None, make_model, (256, 256, 12), 6, log_freq=10, batch_size=batch_size)
    print("Starting Profiling")
    with tf.profiler.experimental.Profile('train/'):
        for batch in ds:
            agent.batch_update(batch)
    while True:
        continue


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    profiling_simple_dqn(100, 32)
