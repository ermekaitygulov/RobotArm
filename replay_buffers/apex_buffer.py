from replay_buffers.cpprb_wrapper import PER
import ray


@ray.remote
class ApeXBuffer():
    def __init__(self, *args, **kwargs):
        self.buffer = PER(*args, **kwargs)

    @ray.method(num_return_vals=2)
    def sample(self, *args, **kwargs):
        batch = self.buffer.sample(*args, **kwargs)
        indexes = batch.pop('indexes')
        return indexes, batch

    def add(self, kwargs):
        self.buffer.add(**kwargs)

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)
