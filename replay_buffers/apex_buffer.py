from collections import deque

import numpy as np
from replay_buffers.sum_tree import SumTree
import ray


@ray.remote
class ApeXBuffer(object):
    def __init__(self, capacity,
                 epsilon=0.001,
                 alpha=0.4,
                 beta=0.6,
                 beta_increment_per_sampling=0.001,
                 n_step=10, gamma=0.99
                 ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.tree = SumTree(capacity)
        self.n_deque = deque([], maxlen=n_step)
        self.gamma = gamma

    def len(self):
        return len(self.tree)

    def full(self):
        return self.tree.full

    def receive(self, transitions, priorities):
        for t, p in zip(transitions, priorities):
            self.tree.add(p, t)

    def sample(self, n):
        idxs = []
        batch = [[] for _ in range(len(self.tree.transition_len))]
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            s = np.random.uniform(0, self.tree.total())
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            idxs.append(idx)
            for b, d in zip(batch, data):
                b.append(d)
        prob = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * prob, -self.beta)
        batch = [np.array(b) for b in batch]
        return idxs, batch, is_weights

    def batch_update(self, tree_idxes, abs_errors):
        errors = [abs_err + self.epsilon for abs_err in abs_errors]
        ps = np.power(errors, self.alpha)
        for ti, p in zip(tree_idxes, ps):
            self.tree.update(ti, p)

    def free_space(self):
        self.tree.free()
