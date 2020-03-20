from collections import deque

import numpy as np
from replay_buffers.sum_tree import SumTree
import ray
import tensorflow as tf


@ray.remote
class PrioritizedBuffer(object):
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

    def __len__(self):
        return len(self.tree)

    def full(self):
        return self.tree.full

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1
        self.tree.add(max_p, transition)  # set the max_p for new transition

    def sample(self, n, normalize_is=False):
        idxs = []
        batch = []
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            s = np.random.uniform(0, self.tree.total())
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            idxs.append(idx)
            batch.append(data)
        prob = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * prob, -self.beta)
        if normalize_is:
            is_weights /= np.max(is_weights)
        # note: b_idx stores indexes in self.tree.tree, not in self.tree.data !!!
        return idxs, batch, is_weights

    def batch_update(self, tree_idxes, abs_errors):
        errors = [abs_err + self.epsilon for abs_err in abs_errors]
        ps = np.power(errors, self.alpha)
        for ti, p in zip(tree_idxes, ps):
            self.tree.update(ti, p)

    def free_space(self):
        self.tree.free()
