from replay_buffers.stable_baselines import PrioritizedReplayBuffer


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
