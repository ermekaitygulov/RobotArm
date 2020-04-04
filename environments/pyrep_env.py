import gym
from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Shape
import numpy as np
from utils.rewards import tolerance


class Rozum(Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, 'Rozum', num_joints=6)
        self.num_joints = 6

    def get_joint_positions_degrees(self):
        angles = [a * 180 / np.pi for a in self.get_joint_positions()]
        return angles

    def set_joint_positions_degrees(self, position):
        angles = [a * np.pi / 180 for a in position]
        self.set_joint_positions(angles)


class RozumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, scene_file='rozum_pyrep.ttt', headless=True):
        self.pr = PyRep()
        self.pr.launch(scene_file, headless=headless)
        self.pr.start()
        self.rozum = Rozum()
        self.gripper = BaxterGripper()
        self.cube = Shape("Cube")
        self.camera = VisionSensor("render")
        self.rozum_tip = self.rozum.get_tip()

        self.action_space = gym.spaces.Box(shape=(self.rozum.num_joints,),
                                           low=-180,
                                           high=180)
        self.observation_space = gym.spaces.Box(shape=self.camera.resolution + [3], low=0, high=255)
        self.reward_range = None
        self.current_step = 0
        self.step_limit = 400
        self.init_angles = self.rozum.get_joint_positions_degrees()
        self.init_cube_pose = self.cube.get_position()
        self.always_render = False
        self.previous_distance = 0

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action: list):
        done = False
        info = None

        position = np.array([j + a for j, a in zip(self.rozum.get_joint_positions_degrees(), action)])
        if all(self.action_space.low < position) and all(position < self.action_space.high):
            self.rozum.set_joint_positions_degrees(position)
            self.pr.step()
        x, y, z = self.rozum_tip.get_position()

        tx, ty, tz = self.cube.get_position()
        curent_distance = np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        reward = tolerance(curent_distance, (0.0, 0.02), 0.1)
        self.current_step += 1
        if self.always_render:
            state = self.render()
        else:
            state = None

        if curent_distance < 0.02:
            done = True
            info = 'SUCCESS'
        elif self.current_step >= self.step_limit:
            done = True
            info = 'FAIL'
        return state, reward, done, info

    def reset(self):
        # self.rozum.set_joint_positions_degrees(self.init_angles)
        # self.cube.set_position(self.init_cube_pose)
        self.pr.stop()
        self.pr.start()
        tx, ty, tz = self.cube.get_position()
        self.cube.set_position([tx + np.random.uniform(-0.2, 0.2), ty, tz])
        state = self.render()
        self.current_step = 0
        return state

    def render(self, mode='human'):
        img = self.camera.capture_rgb()
        img *= 255
        return img.astype('uint8')

    def close(self):
        self.pr.stop()
        self.pr.shutdown()
