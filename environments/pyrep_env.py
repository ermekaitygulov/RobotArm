import gym
from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Shape
import numpy as np
from common.rewards import tolerance


class Rozum(Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, 'Rozum', num_joints=6)
        self.num_joints = 6

    def get_joint_target_positions_degrees(self):
        angles = [a * 180 / np.pi for a in self.get_joint_target_positions()]
        return angles

    def set_joint_target_positions_degrees(self, position):
        angles = [a * np.pi / 180 for a in position]
        self.set_joint_target_positions(angles)


class RozumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, obs_space_keys=('pov', 'angles'), scene_file='rozum_pyrep.ttt',
                 headless=True, always_render=False):
        self.obs_space_keys = (obs_space_keys,) if isinstance(obs_space_keys, str) else obs_space_keys
        self._pyrep = PyRep()
        self._pyrep.launch(scene_file, headless=headless)
        self._pyrep.start()
        self.rozum = Rozum()
        self.gripper = BaxterGripper()
        self.cube = Shape("Cube")
        self.camera = VisionSensor("render")
        self.rozum_tip = self.rozum.get_tip()

        self.action_space = gym.spaces.Box(shape=(self.rozum.num_joints,),
                                           low=-360,
                                           high=360)
        self._available_obs_spaces = dict()
        self._render_dict = dict()
        self._available_obs_spaces['pov'] = gym.spaces.Box(shape=self.camera.resolution + [3],
                                                           low=0, high=255, dtype=np.uint8)
        self._render_dict['pov'] = self.get_image
        self._available_obs_spaces['angles'] = gym.spaces.Box(shape=(self.rozum.num_joints,),
                                                              low=-2 * np.pi, high=2 * np.pi,
                                                              dtype=np.float32)
        self._render_dict['angles'] = self.rozum.get_joint_target_positions
        self._available_obs_spaces['cube'] = gym.spaces.Box(shape=(3,),
                                                              low=0, high=100,
                                                              dtype=np.float32)
        self._render_dict['cube'] = self.cube.get_position
        try:
            if len(self.obs_space_keys) > 1:
                self.observation_space = gym.spaces.Dict({key: self._available_obs_spaces[key]
                                                          for key in self.obs_space_keys})
            else:
                self.observation_space = self._available_obs_spaces[self.obs_space_keys[0]]
        except KeyError as err:
            message = "Observation space {} is not supported.".format(err.args[0])
            message += " Available observation space keys: "
            message += ", ".join(self._available_obs_spaces.keys())
            err.args = (message,)
            raise
        self.reward_range = None
        self.current_step = 0
        self.step_limit = 400
        self.init_angles = self.rozum.get_joint_target_positions_degrees()
        self.init_cube_pose = self.cube.get_position()
        self.always_render = always_render

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action: list):
        done = False
        info = None

        position = np.array([j + a for j, a in zip(self.rozum.get_joint_target_positions_degrees(), action)])
        position = np.clip(position, self.action_space.low, self.action_space.high)
        self.rozum.set_joint_target_positions_degrees(position)
        self._pyrep.step()
        x, y, z = self.rozum_tip.get_position()

        tx, ty, tz = self.cube.get_position()
        curent_distance = np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        reward = tolerance(curent_distance, (0.0, 0.02), 0.25)/25
        self.current_step += 1
        if self.always_render:
            state = self.render()
        else:
            state = None

        if curent_distance < 0.02:
            reward += 10
            done = True
            info = 'SUCCESS'
        elif self.current_step >= self.step_limit:
            done = True
            info = 'FAIL'
        return state, reward, done, info

    def reset(self):
        self._pyrep.stop()
        self._pyrep.start()
        tx, ty, tz = self.cube.get_position()
        self.cube.set_position([tx + np.random.uniform(-0.2, 0.2), ty, tz])
        gripper_open=False
        while not gripper_open:
            gripper_open = self.gripper.actuate(1.0, velocity=0.2)
            self._pyrep.step()
        state = self.render()
        self.current_step = 0
        return state

    def render(self, mode='human'):
        if len(self.obs_space_keys) > 1:
            state = {key: self._render_dict[key]() for key in self.obs_space_keys}
        else:
            state = self._render_dict[self.obs_space_keys[0]]()
        return state

    def get_image(self):
        img = self.camera.capture_rgb()
        img *= 255
        img = img.astype('uint8')
        return img

    def close(self):
        self._pyrep.stop()
        self._pyrep.shutdown()
