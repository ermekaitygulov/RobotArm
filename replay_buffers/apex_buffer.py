from collections import deque

import numpy as np
from replay_buffers.replay_buffers import PrioritizedBuffer
import ray


@ray.remote
class ApeXBuffer(PrioritizedBuffer):
    @ray.method(num_return_vals=3)
    def sample(self, n):
        idxs = []
        batch = {key: [] for key in self.tree.transition_keys}
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            s = np.random.uniform(0, self.tree.total())
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            idxs.append(idx)
            for key in batch.keys():
                batch[key].append(np.array(data[key]))
        prob = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * prob, -self.beta)
        batch = {key: np.array(value) for key, value in batch.items()}
        return idxs, batch, is_weights

    def len(self):
        return len(self.tree)

    def receive(self, transitions, priorities):
        for t, p in zip(transitions, priorities):
            self.tree.add(p, t)
