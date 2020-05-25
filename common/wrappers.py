import os
from collections import deque
from random import random

import cv2
import gym
import numpy as np

from chainerrl.wrappers.atari_wrappers import LazyFrames


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super(FrameSkip, self).__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        obs, done, info = None, None, None
        for _ in range(self._skip):
            _, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        obs = self.env.render()
        return obs, total_reward, done, info


class EpsilonExploration(gym.Wrapper):
    def __init__(self, env, epsilon=0.1, final_epsilon=0.01, epsilon_decay=0.99):
        super(EpsilonExploration, self).__init__(env)
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

    def sample_action(self, action):
        if random() > self.epsilon:
            return action
        else:
            return self.env.sample_action()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.final_epsilon)
        return obs, rew, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order='hwc', stack_key=None):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        """
        super(FrameStack, self).__init__(env)
        self.k = k
        self.observations = deque([], maxlen=k)
        self.stack_axis = {'hwc': 2, 'chw': 0}[channel_order]
        if stack_key:
            space = env.observation_space.spaces.copy()
            low_pov = np.repeat(space[stack_key].low, k, axis=self.stack_axis)
            high_pov = np.repeat(space[stack_key].high, k, axis=self.stack_axis)
            stack_space = gym.spaces.Box(low=low_pov, high=high_pov, dtype=space[stack_key].dtype)
            space[stack_key] = stack_space
            self.observation_space = gym.spaces.Dict(space)
        else:
            stack_space = env.observation_space
            low_pov = np.repeat(stack_space.low, k, axis=self.stack_axis)
            high_pov = np.repeat(stack_space.high, k, axis=self.stack_axis)
            self.observation_space = gym.spaces.Box(low=low_pov, high=high_pov, dtype=stack_space.dtype)
        self.stack_key = stack_key

    def reset(self):
        ob = self.env.reset()
        to_stack = self._to_stack(ob)
        for _ in range(self.k):
            self.observations.append(to_stack)
        return self._get_ob(ob)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        to_stack = self._to_stack(ob)
        self.observations.append(to_stack)
        return self._get_ob(ob), reward, done, info

    def _to_stack(self, ob):
        if self.stack_key:
            to_stack = ob[self.stack_key]
        else:
            to_stack = ob
        return to_stack

    def _get_ob(self, ob):
        assert len(self.observations) == self.k
        if self.stack_key:
            state = ob.copy()
            state[self.stack_key] = LazyFrames(list(self.observations), stack_axis=self.stack_axis)
        else:
            state = LazyFrames(list(self.observations), stack_axis=self.stack_axis)
        return state


class DiscreteWrapper(gym.Wrapper):
    def __init__(self, env, discrete_dict):
        super(DiscreteWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(len(discrete_dict))
        self.discrete_dict = discrete_dict

    def step(self, action):
        s, r, done, info = self.env.step(self.discrete_dict[action])
        return s, r, done, info

    def sample_action(self):
        return self.action_space.sample()


class SaveVideoWrapper(gym.Wrapper):
    current_episode = 0

    def __init__(self, env, path='train/', resize=1, key=None):
        """
        :param env: wrapped environment
        :param path: path to save videos
        :param resize: resize factor
        """
        super(SaveVideoWrapper, self).__init__(env)
        self.path = path
        self.recording = []
        self.rewards = [0]
        self.resize = resize
        self.env.always_render = True
        self.key = key

    def step(self, action):
        """
        make a step in environment
        :param action: agent's action
        :return: observation, reward, done, info
        """
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self._recording_append(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        reset environment and save game video if its not empty
        :param kwargs:
        :return: current observation
        """
        if self.current_episode > 0:
            name = str(self.current_episode).zfill(4) + "r" + str(sum(map(int, self.rewards))).zfill(4) + ".mp4"
            full_path = os.path.join(self.path, name)
            upscaled_video = [self.upscale_image(image, self.resize) for image in self.recording]
            self.save_video(full_path, video=upscaled_video)
        self.current_episode += 1
        self.rewards = [0]
        self.recording = []
        observation = self.env.reset(**kwargs)
        self._recording_append(observation)
        return observation

    def _recording_append(self, observation):
        if self.key:
            pov = self.bgr_to_rgb(observation['pov'])
        else:
            pov = self.bgr_to_rgb(observation)
        self.recording.append(pov)

    @staticmethod
    def upscale_image(image, resize):
        """
        increase image size (for better video quality)
        :param image: original image
        :param resize:
        :return:
        """
        size_x, size_y, size_z = image.shape
        return cv2.resize(image, dsize=(size_x * resize, size_y * resize))

    @staticmethod
    def save_video(filename, video):
        """
        saves video from list of np.array images
        :param filename: filename or path to file
        :param video: [image, ..., image]
        :return:
        """
        size_x, size_y, size_z = video[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (size_x, size_y))
        for image in video:
            out.write(image)
        out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def bgr_to_rgb(image):
        """
        converts BGR image to RGB
        :param image: bgr image
        :return: rgb image
        """
        return image[..., ::-1]


class RozumLogWrapper(gym.Wrapper):
    def __init__(self, env, window, name='agent'):
        super(RozumLogWrapper, self).__init__(env)
        self.accuracy = deque(maxlen=window)
        self.name = name
        self.episodes_done = 0
        self.eps_distances = list()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'grasped' in info.keys():
            self.accuracy.append(info['grasped'])
        self.eps_distances.append(info['distance'])
        if done:
            self.episodes_done += 1
        return observation, reward, done, info

    def reset(self, **kwargs):
        import tensorflow as tf
        observation = self.env.reset(**kwargs)
        if len(self.accuracy) == self.accuracy.maxlen:
            mean = sum(self.accuracy)/len(self.accuracy)
            tf.summary.scalar('accuracy', mean, step=self.episodes_done)
            print('{}_accuracy: {}'.format(self.name, mean))
        if len(self.eps_distances) > 0:
            tf.summary.histogram('distance', self.eps_distances, step=self.episodes_done)
            self.eps_distances = list()
        return observation


class OUExploration(gym.Wrapper):
    def __init__(self, env):
        super(OUExploration, self).__init__(env)
        # TODO tune OUNoise
        self._exploration = OUNoise(env.action_space.shape[0],
                                    env.action_space.low, env.action_space.high)
        self._counter = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._counter += 1
        if done:
            self._counter = 0
            self._exploration.reset()
        return obs, rew, done, info

    def sample_action(self, action):
        return self._exploration.get_action(action, self._counter)


class OUNoise(object):
    def __init__(self, action_dim, low, high, mu=0.0, theta=0.2, max_sigma=0.01, min_sigma=0.001, decay_period=500):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.low = low
        self.high = high
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    noise = OUNoise(1, -0.9, 0.9)
    exploration = list()
    for i in range(1000):
        exploration.append(noise.get_action(0, i))
    plt.plot(exploration)
    plt.show()
