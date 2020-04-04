import random

import numpy as np
from replay_buffers.sum_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def append(self, transition):
        data = transition

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    @property
    def transition_keys(self):
        return self._storage[0].keys()

    def _encode_sample(self, idxes):
        #TODO add ignore keys
        batch = {key: [] for key in self.transition_keys if key != 'q_value'}
        for i in idxes:
            data = self._storage[i]
            for key, value in data.items():
                if key != 'q_value':
                    batch[key].append(np.array(value, copy=False))
        batch = {key: np.array(value) for key, value in batch.items()}
        if 'weights' not in batch.keys():
            batch['weights'] = np.ones_like(batch['rewards'])
        return batch

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def update_priorities(self, *args, **kwargs):
        pass


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha=0.6, eps=1e-6, beta=0.4, beta_increment_per_sampling=0.001):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha
        self._eps = eps
        self._beta = beta
        self._beta_increment = beta_increment_per_sampling

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def append(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().append(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def _sample_asynch(self, batch_size, workers_number=2):
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        prefixsums = np.random.random(batch_size) * every_range_len + np.arange(batch_size) * every_range_len
        res = self._it_sum.find_asynch_prefixsum_idx(prefixsums, workers_number)
        return res

    def sample(self, batch_size, workers_number=2):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        encoded_sample: dict of np.array
            Array of shape(batch_size, ...) and dtype np.*32
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        self._beta = np.min([1., self._beta + self._beta_increment])
        idxes = self._sample_asynch(batch_size, workers_number)
        it_sum = self._it_sum.sum()
        it_min = self._it_min.min()
        p_min = it_min / it_sum
        max_weight = (p_min * len(self._storage)) ** (-self._beta)
        p_sample = np.array([self._it_sum[idx] / it_sum for idx in idxes])
        weights = (p_sample*len(self._storage)) ** (-self._beta)
        weights = weights / max_weight
        encoded_sample = self._encode_sample(idxes)
        encoded_sample['weights'] = weights
        return idxes, encoded_sample

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority + self._eps) ** self._alpha
            self._it_min[idx] = (priority + self._eps) ** self._alpha

            self._max_priority = max(self._max_priority, priority)
