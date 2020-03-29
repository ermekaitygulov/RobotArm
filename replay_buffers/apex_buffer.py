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
        is_weights = np.power(self.tree.n_entries * prob, -self.beta).astype('float32')
        is_weights /= np.max(is_weights)
        batch = {key: np.array(value) for key, value in batch.items()}
        return idxs, batch, is_weights

    def len(self):
        return len(self.tree)

    def receive(self, transitions_and_priorities):
        transitions, priorities = transitions_and_priorities  # pass as 1 argument because of ray
        idxes = [self.tree.add(None, t) for t in transitions]
        self.batch_update(idxes, priorities)
