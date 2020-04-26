from cpprb import PrioritizedReplayBuffer, ReplayBuffer
import numpy as np


class PER(PrioritizedReplayBuffer):
    def __init__(self, state_prefix=('', 'next_', 'n_'), state_keys=('pov', 'angles',),
                 *args, **kwargs):
        super(PER, self).__init__(*args, **kwargs)
        self.state_prefix = state_prefix
        self.state_keys = state_keys

    def sample(self, *args, **kwargs):
        batch = super(PER, self).sample(*args, **kwargs)
        for key, value in batch.items():
            batch[key] = np.squeeze(value)
        for prefix in self.state_prefix:
            batch[prefix+'state'] = {key: batch.pop(prefix+key) for key in self.state_keys}
        return batch

    def add(self, **kwargs):
        for prefix in self.state_prefix:
            state = kwargs.pop(prefix+'state')
            for key, value in state.items():
                kwargs[prefix + key] = value
        super(PER, self).add(**kwargs)

    def get_all_transitions(self):
        batch = super(PER, self).get_all_transitions()
        for prefix in self.state_prefix:
            batch[prefix + 'state'] = {key: batch.pop(prefix + key) for key in self.state_keys}
        return batch


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
