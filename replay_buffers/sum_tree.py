import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.data_pointer = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.data_pointer

    @property
    def transition_keys(self):
        return self.data[0].keys()

    def propagate(self, idx, change):
        parent = (idx - 1) >> 1
        self.tree[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, s):
        """
        find sample on leaf node
        :param idx: index
        :param s: search value
        :return: index of subtree
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        """
        store priority and sample
        :param p:
        :param data:
        :return:
        """
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        if p is not None:
            self.update(idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
            self.full = True
        if self.n_entries < self.capacity:
            self.n_entries += 1
        return idx

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self.propagate(idx, change)

    def get(self, s):
        idx = self.retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def free(self):
        del self.tree
        del self.data
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)
        self.n_entries = 0
        self.data_pointer = 0
        self.full = False
