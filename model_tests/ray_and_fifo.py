import timeit
from model_tests.tf_model_profiling import Dataset
from algorithms.dqn import DQN
from algorithms.model import ClassicCnn, DuelingModel
from numpy import random
import os
import ray

@ray.remote(num_gpus=0.3)
class QueueBuffer:
    def __init__(self, steps=100, batch_size=32):
        import tensorflow as tf
        dtype_list = ['float32',
                      'int32',
                      'float32',
                      'float32',
                      'bool',
                      'float32',
                      'float32',
                      'bool',
                      'float32',
                      'float32']
        names = ['state',
                 'action',
                 'reward',
                 'next_state',
                 'done',
                 'n_state',
                 'n_reward',
                 'n_done',
                 'actual_n',
                 'weights']
        shapes = [(256, 256, 12),
                  (),
                  (),
                  (256, 256, 12),
                  (),
                  (256, 256, 12),
                  (),
                  (),
                  (),
                  ()]
        self.queue = tf.queue.FIFOQueue(100, dtypes=dtype_list, names=names, shapes=shapes)
        self.dataset = Dataset(steps, batch_size)

    def enqueue(self, n):
        _, batch = self.dataset.sample(n)
        self.queue.enqueue_many(batch)

    def dequeue(self, n):
        return self.queue.dequeue_many(n)


@ray.remote(num_gpus=0.3)
class TestAgent(DQN):
    def batch_update(self, batch, start_time):
        _, ntd_loss, _, _ = self.q_network_update(batch['state'], batch['action'],
                                                  batch['reward'], batch['next_state'],
                                                  batch['done'], batch['n_state'],
                                                  batch['n_reward'], batch['n_done'],
                                                  batch['actual_n'], batch['weights'], self.gamma)

        stop_time = timeit.default_timer()
        self._run_time_deque.append(1/(stop_time - start_time))
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


def profiling_asynch_dqn(update_number=100):
    import tensorflow as tf
    tf.config.optimizer.set_jit(True)
    ray.init(webui_host='0.0.0.0', num_gpus=1)
    queue = QueueBuffer.remote()
    agent = TestAgent.remote(None, make_model, (256, 256, 12), 6, log_freq=10)
    tasks = dict()
    tasks[queue.enqueue.remote(32)] = 'enqueue'
    batch = queue.dequeue.remote(32)
    start_time = timeit.default_timer()
    tasks[agent.batch_update.remote(batch, start_time)] = 'updating'
    iteration = 0
    while iteration <= update_number:
        ready_ids, _ = ray.wait(list(tasks))
        first_id = ready_ids[0]
        first = tasks.pop(first_id)
        if first == 'enqueue':
            tasks[queue.enqueue.remote(32)] = 'enqueue'
        elif first == 'updating':
            iteration += 1
            start_time = timeit.default_timer()
            batch = queue.dequeue.remote(32)
            tasks[agent.batch_update.remote(batch, start_time)] = 'updating'


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    profiling_asynch_dqn()


