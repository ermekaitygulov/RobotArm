import os
from collections import deque
from random import random

import cv2
import gym
import numpy as np

from chainerrl.wrappers.atari_wrappers import LazyFrames
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


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


class PopPov(gym.Wrapper):
    def __init__(self, env):
        super(PopPov, self).__init__(env)
        self.observation_space = gym.spaces.Dict({key: value for key, value
                                                  in self.env.observation_space.spaces.items()
                                                  if key != 'pov'})

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs.pop('pov')
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs.pop('pov')
        return obs


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


class CorrelatedExploration(gym.Wrapper):
    def __init__(self, env, mu, sigma, theta=.15, dt=1e-2):
        super(CorrelatedExploration, self).__init__(env)
        self._exploration = OrnsteinUhlenbeckActionNoise(mu, sigma,
                                                         env.action_space.low,
                                                         env.action_space.high,
                                                         theta, dt)
        self._counter = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._counter += 1
        if done:
            self._counter = 0
            self._exploration.reset()
        return obs, rew, done, info

    def sample_action(self, action):
        return self._exploration(action)


class UncorrelatedExploration(CorrelatedExploration):
    def __init__(self, env, mu, sigma):
        super(UncorrelatedExploration, self).__init__(env, mu, sigma, 1., 1.)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, low, high,
                 theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.low = low
        self.high = high
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self, action):
        dx = self.theta * (self.mu - self.x_prev) * self.dt +\
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        x = self.x_prev + dx
        self.x_prev = x
        return np.clip(action + x, self.low, self.high)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class CriticViz(gym.Wrapper):
    def __init__(self, env, online_actor, target_actor, online_critic, target_critic):
        super(CriticViz, self).__init__(env)
        self.online_actor = online_actor
        self.target_actor = target_actor
        self.online_critic = online_critic
        self.target_critic = target_critic
        self.x_axis = np.expand_dims(np.linspace(-2, 2,60), axis=-1)
        self.y_axis = np.expand_dims(np.linspace(-2, 2, 60), axis=-1)
        self.target_values = list()
        self.online_values = list()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        inputs = {key: np.array(value)[None] for key, value in obs.items()}
        online_action = self.online_actor(inputs)[0]
        target_action = self.target_actor(inputs)[0]
        inputs = {key: np.repeat(value, self.x_axis.shape[0], axis=0) for key, value in inputs.items()}
        action_batch = target_action * self.x_axis + online_action * (1 - self.x_axis)
        self.target_values.append(self.target_critic({'state': inputs, 'action': action_batch}))
        self.online_values.append(self.online_critic({'state': inputs, 'action': action_batch}))
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if len(self.target_values) > 0:
            y = np.arange(len(self.target_values))
            self.vis3d(self.target_values, self.online_values, self.x_axis, y)
            self.target_values = list()
            self.online_values = list()
        return obs

    def vis3d(self, target_values, online_values, x_axis, y_axis):
        fig = plt.figure()
        X, Y = np.meshgrid(np.squeeze(x_axis), np.squeeze(y_axis))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        Z = np.squeeze(np.stack(target_values, axis=0))
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        Z = np.squeeze(np.stack(online_values, axis=0))
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

        plt.show()



if __name__ == '__main__':
    noise = OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1, -0.5, 0.5, theta=1., dt=1.)
    exploration = list()
    for i in range(1000):
        exploration.append(noise(0.)*180)
    plt.plot(exploration)
    plt.show()
