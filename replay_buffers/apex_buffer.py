from replay_buffers.cpprb_wrapper import PER
import ray


class ApeXBuffer(PER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, kwargs):
        super().add(**kwargs)

    def update_priorities(self, kwargs):
        super().update_priorities(**kwargs)
