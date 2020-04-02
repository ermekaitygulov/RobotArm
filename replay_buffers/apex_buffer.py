from replay_buffers.replay_buffers import PrioritizedReplayBuffer
import ray


@ray.remote
class ApeXBuffer(PrioritizedReplayBuffer):
    @ray.method(num_return_vals=2)
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs)

    def len(self):
        return self.__len__()

    def append(self, transition, priority):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super(PrioritizedReplayBuffer, self).append(transition)
        self._it_sum[idx] = priority ** self._alpha
        self._it_min[idx] = priority ** self._alpha

    def receive_batch(self, transition_and_priorities):
        for t, p in zip(*transition_and_priorities):
            self.append(t, p)
