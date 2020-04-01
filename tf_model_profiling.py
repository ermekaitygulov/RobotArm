import timeit

from algorithms.dqn import DQN
from algorithms.model import ClassicCnn, DuelingModel
from numpy import random
import os
import ray


class TestBuffer:
    def __init__(self):
        pass

    def sample(self, batch_size):
        minibatch = dict()
        idx = None
        minibatch['state'] = random.randint(0, 255, size=(batch_size, 256, 256, 12), dtype='uint8')
        minibatch['action'] = random.randint(0, 5, size=(batch_size))
        minibatch['reward'] = random.randint(0, 10, size=(batch_size))
        minibatch['next_state'] = random.randint(0, 255, size=(batch_size, 256, 256, 12), dtype='uint8')
        minibatch['done'] = random.randint(0, 1, size=(batch_size))
        minibatch['n_state'] = random.randint(0, 255, size=(batch_size, 256, 256, 12), dtype='uint8')
        minibatch['n_reward'] = random.randint(0, 10, size=(batch_size))
        minibatch['n_done'] = random.randint(0, 1, size=(batch_size))
        minibatch['actual_n'] = random.randint(0, 5, size=(batch_size))
        minibatch['weights'] = random.uniform(size=[batch_size])
        return idx, minibatch

    def update_priorities(self, *args, **kwargs):
        pass

@ray.remote(num_gpus=0, num_cpus=2)
class QueueBuffer:
    def __init__(self):
        import tensorflow as tf
        dtype_list = ('float32',
                      'int32',
                      'float32',
                      'float32',
                      'bool',
                      'float32',
                      'float32',
                      'bool',
                      'float32',
                      'float32')
        names = ('state',
                 'action',
                 'reward',
                 'next_state',
                 'done',
                 'n_state',
                 'n_reward',
                 'n_done',
                 'actual_n'
                 'weights')
        self.queue = tf.queue.FIFOQueue(500, dtypes=dtype_list, names=names)
        self.buffer = TestBuffer()

    def enqueue(self, n):
        _, batch = self.buffer.sample(n)
        self.queue.enqueue_many(batch)

    def dequeue(self, n):
        return self.queue.dequeue_many(n)


@ray.remote(num_gpus=0.3)
class TestAgent(DQN):
    def batch_update(self, batch):
        start_time = timeit.default_timer()
        _, ntd_loss, _, _ = self.q_network_update(batch['state'], batch['action'],
                                                  batch['reward'], batch['next_state'],
                                                  batch['done'], batch['n_state'],
                                                  batch['n_reward'], batch['n_done'],
                                                  batch['actual_n'], batch['is_weights'], self.gamma)

        stop_time = timeit.default_timer()
        self._run_time_deque.append(stop_time - start_time)
        self.schedule()
        return ntd_loss


def make_model(name, input_shape, output_shape):
    import tensorflow as tf
    from utils.util import config_gpu
    config_gpu()
    base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])
    head = DuelingModel([1024], output_shape)
    model = tf.keras.Sequential([base, head], name)
    model.build((None, ) + input_shape)
    return model


def profiling_simple_dqn(update_number=100):
    import tensorflow as tf
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

    replay_buffer = TestBuffer()

    agent = DQN(replay_buffer, make_model, (256, 256, 12), 6, log_freq=10)
    print("Starting Profiling")
    with tf.profiler.experimental.Profile('train/'):
        agent.update(update_number)

    while True:
        continue

def profiling_asynch_dqn(update_number=100):
    import tensorflow as tf
    tf.config.optimizer.set_jit(True)
    ray.init(webui_host='0.0.0.0', num_gpus=1)
    queue = QueueBuffer.remote()
    agent = TestAgent.remote(None, make_model, (256, 256, 12), 6, log_freq=10)
    tasks = dict()
    tasks[queue.enqueue.remote(32)] = 'enqueue'
    batch = queue.dequeue.remote(32)
    tasks[agent.batch_update.remote(batch)] = 'updating'
    iteration = 0
    with tf.profiler.experimental.Profile('train/'):
        while iteration <= update_number:
            ready_ids, _ = ray.wait(list(tasks))
            first_id = ready_ids[0]
            first = tasks.pop(first_id)
            if first == 'enqueue':
                tasks[queue.enqueue.remote(32)] = 'enqueue'
            elif first == 'updating':
                batch = queue.dequeue.remote(32)
                tasks[agent.batch_update.remote(batch)] = 'updating'

    while True:
        continue

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    profiling_asynch_dqn()
