import gym
from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Shape
import numpy as np
from common.rewards import tolerance
import tensorflow as tf


class Rozum(Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, 'Rozum', num_joints=6)
        self.num_joints = 6


class RozumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, obs_space_keys=('pov', 'arm'), scene_file='rozum_pyrep.ttt',
                 headless=True):
        self.obs_space_keys = (obs_space_keys,) if isinstance(obs_space_keys, str) else obs_space_keys
        self._pyrep = PyRep()
        self._pyrep.launch(scene_file, headless=headless)
        self._pyrep.start()
        self.rozum = Rozum()
        self.gripper = BaxterGripper()
        self.cube = Shape("Cube")
        self.graspable_objects = [self.cube, ]
        self.camera = VisionSensor("render")
        self.rozum_tip = self.rozum.get_tip()

        low = np.array([-0.5 for _ in range(self.rozum.num_joints)] + [0., ])
        high = np.array([0.5 for _ in range(self.rozum.num_joints)] + [1., ])
        self.angles_scale = np.array([np.pi for _ in range(self.rozum.num_joints)])
        self.action_space = gym.spaces.Box(low=low,
                                           high=high)
        self._available_obs_spaces = dict()
        self._render_dict = dict()
        self._available_obs_spaces['pov'] = gym.spaces.Box(shape=self.camera.resolution + [3],
                                                           low=0, high=255, dtype=np.uint8)
        self._render_dict['pov'] = self.get_image
        low = np.array([-angle for angle in self.angles_scale] + [0., ])
        high = np.array([angle for angle in self.angles_scale] + [1., ])
        self._available_obs_spaces['arm'] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._render_dict['arm'] = self.get_arm_state
        self._available_obs_spaces['cube'] = gym.spaces.Box(shape=(7,),
                                                            low=0, high=100,
                                                            dtype=np.float32)
        self._render_dict['cube'] = self.cube.get_pose
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
        self.step_limit = 1000
        self.init_angles = self.rozum.get_joint_positions()
        self.init_cube_pose = self.cube.get_pose()
        self._eps_done = 0

    def get_arm_state(self):
        arm = self.rozum.get_joint_positions()
        arm.append(self.gripper.get_open_amount()[0])
        return arm

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action: list):
        done = False
        info = None
        joint_action, ee_action = action[:-1], action[-1]
        current_ee = (1.0 if self.gripper.get_open_amount()[0] > 0.9
                      else 0.0)
        grasped = False
        if ee_action > 0.5:
            ee_action = 1.0
        elif ee_action < 0.5:
            ee_action = 0.0
        if current_ee != ee_action:
            gripper_done = False
            while not gripper_done:
                gripper_done = self.gripper.actuate(ee_action, velocity=0.2)
                self._pyrep.step()
            if ee_action == 0.0:
                # If gripper close action, the check for grasp.
                for g_obj in self.graspable_objects:
                    grasped = self.gripper.grasp(g_obj)
            self.current_step += 1
        else:
            joint_action *= self.angles_scale
            position = [j + a for j, a in zip(self.rozum.get_joint_positions(), joint_action)]
            self.rozum.set_joint_target_positions(position)
            current_pose = self.rozum.get_joint_positions()
            step = 0
            while True:
                previous_pose = current_pose.copy()
                self._pyrep.step()
                step += 1
                current_pose = self.rozum.get_joint_positions()
                block_case = all([abs(c-p) < 0.015 for c, p in zip(current_pose, previous_pose)])
                done_case = all([abs(c-t) < 0.015 for c, t in zip(current_pose, position)])
                if block_case or done_case or step > 20:
                    break
            self.current_step += step
        x, y, z = self.rozum_tip.get_position()

        tx, ty, tz = self.cube.get_position()
        current_distance = np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        reward = tolerance(current_distance, (0.0, 0.06), 0.25)/25
        state = self.render()

        if grasped:
            reward += 20
            done = True
            info = 'SUCCESS'
        elif self.current_step >= self.step_limit:
            done = True
            info = 'FAIL'
        if done:
            self._eps_done += 1
            tf.summary.scalar('final_distance', current_distance, step=self._eps_done)
            tf.summary.flush()
        return state, reward, done, info

    def reset(self):
        self._pyrep.stop()
        self._pyrep.start()
        pose = self.init_cube_pose.copy()
        pose[0] += np.random.uniform(-0.05, 0.05)
        pose[1] += np.random.uniform(-0.1, 0.1)
        self.cube.set_pose(pose)
        # self.cube.set_color([0.,np.random.uniform(0., 255.),0.])
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
