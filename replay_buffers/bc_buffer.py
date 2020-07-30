from replay_buffers.stable_baselines import PrioritizedReplayBuffer
from copy import deepcopy
import numpy as np


class DQfDBuffer(PrioritizedReplayBuffer):
    def __init__(self, demo_eps=1., *args, **kwargs):
        super(DQfDBuffer, self).__init__(*args, **kwargs)
        self._demo_eps = demo_eps
        self._demo_border = 0

    def add_single_transition(self, priority, **kwargs):
        self._next_idx = max(self._next_idx, self._demo_border)
        super(DQfDBuffer, self).add_single_transition(priority, **kwargs)

    def add_demo(self, **kwargs):
        if self._next_idx > self._demo_border:
            self._next_idx = 0
        super(DQfDBuffer, self).add_single_transition(self._max_priority, **kwargs)
        self._demo_border = min(self._demo_border + 1, self._maxsize)

    def update_priorities(self, indexes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        indexes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(indexes) == len(priorities)
        for idx, priority in zip(indexes, priorities):
            assert priority >= 0
            assert 0 <= idx < len(self._storage)
            if idx > self._demo_border:
                self._it_sum[idx] = (priority + self._eps) ** self._alpha
                self._it_min[idx] = (priority + self._eps) ** self._alpha
                self._max_priority = max(self._max_priority, priority)
            else:
                self._it_sum[idx] = (priority + self._demo_eps) ** self._alpha
                self._it_min[idx] = (priority + self._demo_eps) ** self._alpha


class AggregatedBuff:
    def __init__(self, base, steps_to_decay=50):
        self.demo_buff = deepcopy(base)
        self.replay_buff = base
        self.steps_to_decay = steps_to_decay

    def add(self, **kwargs):
        self.replay_buff.add(**kwargs)
        if self.demo_buff and self.get_stored_size() > self.steps_to_decay:
            try:
                self.demo_buff.clear()
            except AttributeError:
                del self.demo_buff
            self.demo_buff = None

    def add_demo(self, **kwargs):
        self.demo_buff.add(**kwargs)

    @property
    def proportion(self):
        if self.steps_to_decay == 0 or self.demo_buff is None:
            proportion = 1.
        else:
            proportion = min(1., self.get_stored_size() / self.steps_to_decay)
        return proportion

    def sample(self, n=32, beta=0.4):
        agent_n = int(n*self.proportion)
        demo_n = n - agent_n
        if demo_n > 0 and agent_n > 0:
            demo_samples = self.demo_buff.sample(demo_n, beta)
            replay_samples = self.replay_buff.sample(agent_n, beta)
            demo_samples['indexes'] += 1
            demo_samples['indexes'] *= -1
            samples = {key: np.concatenate((replay_samples[key], demo_samples[key]))
                       for key in replay_samples.keys()}
        elif agent_n == 0:
            samples = self.demo_buff.sample(demo_n, beta)
            samples['indexes'] += 1
            samples['indexes'] *= -1
        else:
            samples = self.replay_buff.sample(agent_n, beta)
        samples = {key: np.squeeze(value) for key, value in samples.items()}
        return samples

    def update_priorities(self, indexes, priorities):
        if self.demo_buff:
            demo_indexes = indexes < 0
            self.demo_buff.update_priorities(indexes[demo_indexes] * (-1) - 1, priorities[demo_indexes])
        replay_indexes = indexes >= 0
        self.replay_buff.update_priorities(indexes[replay_indexes], priorities[replay_indexes])

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.replay_buff, name)
