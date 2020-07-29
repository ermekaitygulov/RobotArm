import pickle
import os
from gym.spaces.dict import Dict


class DataLoader:
    def __init__(self, path, obs_keys=('pov',), number_of_transitions=None):
        self.path = path
        file = [f for f in os.listdir(self.path) if f.endswith(".pkl")][0]
        sample_data = self.load_obj(os.path.join(self.path, file))
        self.action_space = sample_data['acs_space']
        if isinstance(sample_data['obs_space'], Dict):
            if len(obs_keys) == 1:
                self.observation_space = sample_data['obs_space'][obs_keys[0]]
            else:
                self.observation_space = Dict({key: space for key, space in sample_data['obs_space'].spaces.items()
                                               if key in obs_keys})
            self.obs_keys = obs_keys
        else:
            self.observation_space = sample_data['obs_space']
            self.obs_keys = None
        self.number_of_transitions = number_of_transitions

    def sarsd_iter(self):
        transitions_add = 0
        for file in os.listdir(self.path):
            if file.endswith(".pkl"):
                data = self.load_obj(os.path.join(self.path, file))
                observations = data['observation']
                if self.obs_keys:
                    if len(self.obs_keys) == 1:
                        observations = [o[self.obs_keys[0]] for o in observations]
                    else:
                        observations = [{key: data for key, data in o.items()
                                        if key in self.obs_keys} for o in observations]
                for transition in zip(observations[:-1], data['action'], data['reward'],
                                      observations[1:], data['done']):
                    transitions_add += 1
                    yield transition
                    if self.number_of_transitions and transitions_add > self.number_of_transitions:
                        break
                if self.number_of_transitions and transitions_add > self.number_of_transitions:
                    break


    @staticmethod
    def load_obj(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
