from collections import deque

import gym
import numpy as np

from chainerrl.wrappers.atari_wrappers import LazyFrames


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        obs, done, info = None, None, None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order='hwc', use_tuple=False):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.observations = deque([], maxlen=k)
        self.stack_axis = {'hwc': 2, 'chw': 0}[channel_order]
        self.use_tuple = use_tuple

        pov_space = env.observation_space
        low_pov = np.repeat(pov_space.low, k, axis=self.stack_axis)
        high_pov = np.repeat(pov_space.high, k, axis=self.stack_axis)
        self.observation_space = gym.spaces.Box(low=low_pov, high=high_pov, dtype=pov_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.observations.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.observations.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.observations) == self.k
        return LazyFrames(list(self.observations), stack_axis=self.stack_axis)


class DiscreteWrapper(gym.Wrapper):
    def __init__(self, env, discrete_dict):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(len(discrete_dict))
        self.discrete_dict = discrete_dict

    def step(self, action):
        s, r, done, info = self.env.step(self.discrete_dict[action])
        return s, r, done, info

    def sample_action(self):
        return self.action_space.sample()
