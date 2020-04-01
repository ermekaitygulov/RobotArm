from algorithms.dqn import DQN
from algorithms.model import ClassicCnn, DuelingModel
import tensorflow as tf
from numpy import random
import os

class TestBuffer:
    def __init__(self):
        pass

    def sample(self, batch_size):
        idxes = None
        weights = random.uniform(size=[batch_size]).astype('float32')
        minibatch = dict()
        minibatch['state'] = random.randint(0, 255, size=(batch_size, 12, 256, 256), dtype='uint8')
        minibatch['action'] = random.randint(0, 5, size=(batch_size))
        minibatch['reward'] = random.randint(0, 10, size=(batch_size))
        minibatch['next_state'] = random.randint(0, 255, size=(batch_size, 12, 256, 256), dtype='uint8')
        minibatch['done'] = random.randint(0, 1, size=(batch_size))
        minibatch['n_state'] = random.randint(0, 255, size=(batch_size, 12, 256, 256), dtype='uint8')
        minibatch['n_reward'] = random.randint(0, 10, size=(batch_size))
        minibatch['n_done'] = random.randint(0, 1, size=(batch_size))
        minibatch['actual_n'] = random.randint(0, 5, size=(batch_size))
        return idxes, minibatch, weights

    def update_priorities(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    tf.debugging.set_log_device_placement(False)

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

    replay_buffer = TestBuffer()

    def make_model(name, obs_shape, action_shape):
        base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2], data_format='channels_first')
        head = DuelingModel([1024], action_shape)
        model = tf.keras.Sequential([base, head], name)
        model.build((None, ) + obs_shape)
        return model
    agent = DQN(replay_buffer, make_model, (12, 256, 256), 6, log_freq=10)
    print("Starting Profiling")
    with tf.profiler.experimental.Profile('train/'):
        agent.update(100)

    while True:
        continue
