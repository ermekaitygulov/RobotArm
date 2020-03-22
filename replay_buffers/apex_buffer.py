from collections import deque

import numpy as np
from replay_buffers.replay_buffers import PrioritizedBuffer
import ray


@ray.remote
class ApeXBuffer(PrioritizedBuffer):
    def len(self):
        return len(self.tree)

    def receive(self, transitions, priorities):
        for t, p in zip(transitions, priorities):
            self.tree.add(p, t)
