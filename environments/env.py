import time
import numpy as np
import cv2
import gym
from vrep.robots import Rozum


class RozumEnv:

    def __init__(self, robot_run_file='coppeliaSim.sh', robot_port=19999):
        self.rozum = Rozum(robot_run_file, robot_port)
        self.action_range = [-5, 5]
        self.action_dim = self.rozum.DoF

        self.action_space = gym.spaces.Box(shape=(self.action_dim,), low=-5, high=5)
        self.observation_space = gym.spaces.Box(shape=(3 + self.action_dim * 2,), low=-180, high=180)
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]

        self.init_angles = [joint.value for joint in self.rozum.joints]
        self.init_cube_pose = self.rozum.cube.value
        self.init_goal_pose = self.rozum.goal.value

    def sample_action(self):
        return np.random.uniform(*self.rozum.action_bound, size=self.action_dim)

    def step(self, action):
        for i in range(self.action_dim):
            self.rozum.joints[i].value += action[i]
        time.sleep(0.3)
        pose = self.rozum.tip.value
        r = 0.0
        done = False
        target = self.rozum.cube.value
        s = self.rozum.side_cam.value
        d = np.linalg.norm(pose - target)
        r += (-d - 0.01 * np.square(action).sum())
        if d < 0.02:
            done = True
        return s, r, done, None

    def reset(self):
        for i in range(self.action_dim):
            self.rozum.joints[i].value = self.init_angles[i]
        self.rozum.open_gripper()
        self.rozum.cube.value = self.init_cube_pose
        self.rozum.goal.value = self.init_goal_pose
        time.sleep(2)
        s = self.rozum.side_cam.value
        return s

    def render(self):
        img = self.rozum.side_cam.value
        return img

    def close(self):
        self.rozum.stop_simulation()
        self.rozum.disconnect()


if __name__ == '__main__':
    env = RozumEnv()
    a = env.sample_action()
    env.step(a)
    a = env.reset()
    env.close()