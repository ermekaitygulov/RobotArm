from cpprb import PrioritizedReplayBuffer, ReplayBuffer
import timeit
from collections import deque
import numpy as np


class PER(PrioritizedReplayBuffer):
    def __init__(self, state_prefix=('', 'next_', 'n_'), state_keys=('pov', 'angles',),
                 *args, **kwargs):
        super(PER, self).__init__(*args, **kwargs)
        self.state_prefix = state_prefix
        self.state_keys = state_keys
        self.add_deque = deque(maxlen=10)
        self.sample_deque = deque(maxlen=10)
        self.update_deque = deque(maxlen=10)

    def sample(self, *args, **kwargs):
        start_time = timeit.default_timer()
        batch = super(PER, self).sample(*args, **kwargs)
        for key, value in batch.items():
            batch[key] = np.squeeze()
            if 'pov' in key:
                batch[key] /= 255
        for prefix in self.state_prefix:
            batch[prefix+'state'] = {key: batch.pop(prefix+key) for key in self.state_keys}
        stop_time = timeit.default_timer()
        self.sample_deque.append(stop_time - start_time)
        print("Sample time (it/sec): {:.2f}".format(len(self.sample_deque)/sum(self.sample_deque)))
        return batch

    def add(self, **kwargs):
        start_time = timeit.default_timer()
        for prefix in self.state_prefix:
            state = kwargs.pop(prefix+'state')
            for key, value in state.items():
                kwargs[prefix + key] = value
        super(PER, self).add(**kwargs)
        stop_time = timeit.default_timer()
        self.add_deque.append(stop_time - start_time)
        print("Add time (it/sec): {:.2f}".format(len(self.add_deque) / sum(self.add_deque)))

    def get_all_transitions(self):
        batch = super(PER, self).get_all_transitions()
        for prefix in self.state_prefix:
            batch[prefix + 'state'] = {key: batch.pop(prefix + key) for key in self.state_keys}
        return batch

    def update_priorities(self, *args, **kwargs):
        start_time = timeit.default_timer()
        super(PER, self).update_priorities(*args, **kwargs)
        stop_time = timeit.default_timer()
        self.update_deque.append(stop_time - start_time)
        print("Update time (it/sec): {:.2f}".format(len(self.update_deque) / sum(self.update_deque)))


class RB(ReplayBuffer):
    def __init__(self, state_prefix=('', 'next_', 'n_'), state_keys=('pov', 'angles',),
                 *args, **kwargs):
        super(RB, self).__init__(*args, **kwargs)
        self.state_prefix = state_prefix
        self.state_keys = state_keys

    def sample(self, *args, **kwargs):
        batch = super(RB, self).sample(*args, **kwargs)
        for prefix in self.state_prefix:
            batch[prefix+'state'] = {key: batch.pop(prefix+key) for key in self.state_keys}
        return batch

    def add(self, **kwargs):
        for prefix in self.state_prefix:
            state = kwargs.pop(prefix+'state')
            for key, value in state.items():
                kwargs[prefix + key] = value
        super(RB, self).add(**kwargs)

    def get_all_transitions(self):
        batch = super(RB, self).get_all_transitions()
        for prefix in self.state_prefix:
            batch[prefix + 'state'] = {key: batch.pop(prefix + key) for key in self.state_keys}
        return batch
