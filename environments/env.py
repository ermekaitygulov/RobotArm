import time
import numpy as np
import gym
from vrep.robots import Rozum


class RozumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, robot_run_file='coppeliaSim.sh', robot_port=19999):
        self.rozum = Rozum(robot_run_file, robot_port)
        self.action_range = [-5, 5]

        self.action_space = gym.spaces.Box(shape=(self.rozum.DoF,),
                                           low=self.rozum.action_bound[0],
                                           high=self.rozum.action_bound[1])
        self.observation_space = gym.spaces.Box(shape=self.rozum.side_cam_dim, low=0, high=255)
        self.reward_range = None
        self.current_step = 0
        self.step_limit = 300
        self.init_angles = [joint.value for joint in self.rozum.joints]
        self.init_cube_pose = self.rozum.cube.value
        self.init_goal_pose = self.rozum.goal.value

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action: list):
        for i in range(self.rozum.DoF):
            self.rozum.joints[i].value += action[i]
        # time.sleep(0.3)
        pose = self.rozum.tip.value
        done = False
        target = self.rozum.cube.value
        s = self.rozum.side_cam.value
        d = np.linalg.norm(pose - target)
        r = (-d - 0.01 * np.square(action).sum())
        self.current_step += 1
        if d < 0.02 or self.current_step >= self.step_limit:
            done = True
        return s, r, done, None

    def reset(self):
        for i in range(self.rozum.DoF):
            self.rozum.joints[i].value = self.init_angles[i]
        self.rozum.open_gripper()
        self.rozum.cube.value = self.init_cube_pose
        self.rozum.goal.value = self.init_goal_pose
        # time.sleep(2)
        s = self.rozum.side_cam.value
        self.current_step = 0
        return s

    def render(self, mode='human'):
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
