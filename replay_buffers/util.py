import numpy as np
import gym


class DictWrapper:
    def __init__(self, replay_buffer, state_prefix=('', 'next_', 'n_'), state_keys=('pov', 'angles',)):
        self.replay_buffer = replay_buffer
        self.state_prefix = state_prefix
        self.state_keys = state_keys

    def sample(self, *args, **kwargs):
        batch = self.replay_buffer.sample(*args, **kwargs)
        for key, value in batch.items():
            batch[key] = np.squeeze(value)
        for prefix in self.state_prefix:
            batch[prefix+'state'] = {key: batch.pop(prefix+key) for key in self.state_keys}
        return batch

    def add(self, **kwargs):
        for prefix in self.state_prefix:
            state = kwargs.pop(prefix+'state')
            for key, value in state.items():
                kwargs[prefix + key] = value
        self.replay_buffer.add(**kwargs)

    def get_all_transitions(self):
        batch = self.replay_buffer.get_all_transitions()
        for prefix in self.state_prefix:
            batch[prefix + 'state'] = {key: batch.pop(prefix + key) for key in self.state_keys}
        return batch

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.replay_buffer, name)


def get_dtype_dict(observation_space, action_space):
    action_shape = action_space.shape
    action_shape = action_shape if len(action_shape) > 0 else 1
    action_dtype = action_space.dtype
    action_dtype = 'int32' if np.issubdtype(action_dtype, int) else action_dtype
    action_dtype = 'float32' if np.issubdtype(action_dtype, float) else action_dtype
    env_dict = {'action': {'shape': action_shape,
                           'dtype': action_dtype},
                'reward': {'dtype': 'float32'},
                'done': {'dtype': 'bool'},
                'n_reward': {'dtype': 'float32'},
                'n_done': {'dtype': 'bool'},
                'actual_n': {'dtype': 'float32'}
                }
    dtype_dict = {key: value['dtype'] for key, value in env_dict.items()}
    for prefix in ('', 'next_', 'n_'):
        if isinstance(observation_space, gym.spaces.Dict):
            dtype_dict[prefix+'state'] = dict()
            for name, space in observation_space.spaces.items():
                env_dict[prefix + name] = {'shape': space.shape,
                                           'dtype': space.dtype}
                dtype_dict[prefix+'state'][name] = space.dtype
        else:
            env_dict[prefix + 'state'] = {'shape': observation_space.shape,
                                          'dtype': observation_space.dtype}
            dtype_dict[prefix + 'state'] = observation_space.dtype
    dtype_dict.update(weights='float32', indexes='int32')
    return env_dict, dtype_dict
