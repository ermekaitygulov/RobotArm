import gym
from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Shape
import numpy as np


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


class RG2(Gripper):
    def __init__(self, count: int = 0):
        super().__init__(count, 'RG2',
                         ['RG2_rightJoint0', 'RG2_leftJoint0',
                          'RG2_rightJoint1', 'RG2_leftJoint1',
                          'RG2_rightJoint2', 'RG2_leftJoint2',
                          'RG2_rightJoint3', 'RG2_leftJoint3',
                          'RG2_rightJoint4', 'RG2_leftJoint4'])


class RozumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch('rozum_pyrep.ttt', headless=True)
        self.pr.start()
        self.action_range = [-5, 5]
        self.rozum = Rozum()
        self.gripper = RG2()
        self.cube = Shape("Cube")
        self.camera = VisionSensor("render")
        self.rozum_tip = self.rozum.get_tip()

        self.action_space = gym.spaces.Box(shape=(self.rozum.num_joints,),
                                           low=-180,
                                           high=180)
        self.observation_space = gym.spaces.Box(shape=self.camera.resolution, low=0, high=255)
        self.reward_range = None
        self.current_step = 0
        self.step_limit = 4000
        self.init_angles = self.rozum.get_joint_positions_degrees()
        self.init_cube_pose = self.cube.get_position()

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action: list):
        position = [j + a for j, a in zip(self.rozum.get_joint_positions_degrees(), action)]
        self.rozum.set_joint_positions_degrees(position)
        self.pr.step()
        x, y, z = self.rozum_tip.get_position()
        done = False
        tx, ty, tz = self.cube.get_position()
        state = self.render()
        reward = -np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        self.current_step += 1
        if reward < 0.02 or self.current_step >= self.step_limit:
            done = True
        return state, reward, done, None

    def reset(self):
        self.rozum.set_joint_positions_degrees(self.init_angles)
        self.cube.set_position(self.init_cube_pose)
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
