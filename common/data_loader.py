import pickle
import os


class DataLoader:
    def __init__(self, path):
        self.path = path
        file = [f for f in os.listdir(self.path) if f.endswith(".pkl")][0]
        sample_data = self.load_obj(file)
        self.observation_space = sample_data['obs_space']
        self.action_space = sample_data['acs_space']

    def sarsd_iter(self):
        for file in os.listdir(self.path):
            if file.endswith(".pkl"):
                data = self.load_obj(os.path.join(self.path, file))
                for transition in zip(data['observation'][:-1], data['action'], data['reward'],
                                      data['observation'][1:], data['done']):
                    yield transition


    @staticmethod
    def load_obj(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
