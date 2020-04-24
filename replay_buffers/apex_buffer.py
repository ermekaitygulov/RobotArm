from replay_buffers.cpprb_wrapper import PER
import ray


@ray.remote
class ApeXBuffer(PER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @ray.method(num_return_vals=2)
    def sample(self, *args, **kwargs):
        batch = super().sample(*args, **kwargs)
        indexes = batch.pop('indexes')
        return indexes, batch

    def add(self, kwargs):
        super().add(**kwargs)
