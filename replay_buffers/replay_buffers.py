from collections import deque

import numpy as np
from replay_buffers.sum_tree import SumTree


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

    def append(self, transition):
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


class AggregatedBuff:
    def __init__(self, capacity, demo_n=32):
        self.demo_buff = PrioritizedBuffer(capacity=capacity, epsilon=1.0)
        self.replay_buff = PrioritizedBuffer(capacity=capacity)
        self.demo_n = demo_n
        self.no_demo = False
        self.capacity = capacity

    def store(self, transition, demo=False):
        if demo:
            self.demo_buff.store(transition)
        else:
            self.replay_buff.store(transition)

    def sample(self, n=32, proportion=0.0):
        idxs, batch, is_weights = [], [], ()
        agent_n = int(n*proportion)
        self.demo_n = n - agent_n
        if self.demo_n != 0:
            demo_idxs, demo_batch, demo_is_weights = self.demo_buff.sample(self.demo_n)
            idxs += demo_idxs
            batch += demo_batch
            is_weights += (demo_is_weights,)
        if agent_n > 0:
            replay_idxs, replay_batch, replay_is_weights = self.replay_buff.sample(agent_n)
            idxs += replay_idxs
            batch += replay_batch
            is_weights += (replay_is_weights,)
        is_weights = np.concatenate(is_weights)
        is_weights /= np.max(is_weights)
        return idxs, batch, is_weights

    def batch_update(self, tree_idxes, abs_errors):
        demo_errors = abs_errors[:self.demo_n]
        replay_errors = abs_errors[self.demo_n:]
        if not isinstance(self.demo_buff, list):
            self.demo_buff.batch_update(tree_idxes[:self.demo_n], demo_errors)
        self.replay_buff.batch_update(tree_idxes[self.demo_n:], replay_errors)

    def __len__(self):
        return len(self.replay_buff)

    def full(self):
        return {'demo': self.demo_buff.full(), 'replay': self.replay_buff.full()}

    def free_demo(self):
        self.demo_buff.free_space()


class DQfDBuff(PrioritizedBuffer):
    def __init__(self, capacity,
                 epsilon=0.001,
                 alpha=0.4,
                 beta=0.6,
                 beta_increment_per_sampling=0.001,
                 demo_epsilon=1.0):
        super(DQfDBuff, self).__init__(capacity, epsilon, alpha,
                                       beta, beta_increment_per_sampling)
        self.demo_epsilon = demo_epsilon
        self.agent_pointer = 0
        self.demo_pointer = 0

    def batch_update(self, tree_idxes, abs_errors):
        data_idxes = [idx - self.tree.capacity + 1 for idx in tree_idxes]
        epsilon = [self.epsilon if idx > self.demo_pointer
                   else self.demo_epsilon for idx in data_idxes]
        errors = [abs_err + eps for abs_err, eps in zip(abs_errors, epsilon)]
        ps = np.power(errors, self.alpha)
        for ti, p in zip(tree_idxes, ps):
            self.tree.update(ti, p)

    def store(self, transition, demo=False):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1
        if demo:
            self.tree.data_pointer = self.demo_pointer
            self.demo_pointer += 1
            if self.demo_pointer >= self.tree.capacity:
                self.demo_pointer = 0
        else:
            if self.agent_pointer >= self.tree.capacity or self.agent_pointer <= self.demo_pointer:
                self.agent_pointer = self.demo_pointer + 1
            self.tree.data_pointer = self.agent_pointer
            self.agent_pointer += 1

        self.tree.add(max_p, transition)
