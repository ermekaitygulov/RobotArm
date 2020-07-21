import os

import cv2
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


class RozumEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, obs_space_keys=('pov', 'arm'), scene_file='rozum_pyrep.ttt',
                 headless=True, video_path=None, pose_sigma=20):
        self.obs_space_keys = (obs_space_keys,) if isinstance(obs_space_keys, str) else obs_space_keys
        # PyRep
        self._pyrep = PyRep()
        self._pyrep.launch(scene_file, headless=headless)
        self._pyrep.start()
        self.rozum = Rozum()
        self.rozum.set_control_loop_enabled(True)
        self.gripper = BaxterGripper()
        self.gripper.set_control_loop_enabled(True)
        self._initial_robot_state = (self.rozum.get_configuration_tree(),
                                     self.gripper.get_configuration_tree())
        self.cube = Shape("Cube")
        self.graspable_objects = [self.cube, ]
        self.camera = VisionSensor("render")
        self.rozum_tip = self.rozum.get_tip()

        # Action and Observation spaces
        self.angles_scale = np.array([np.pi for _ in range(self.rozum.num_joints)])
        low = np.array([-20/180 for _ in range(self.rozum.num_joints)] + [0., ])
        high = np.array([20/180 for _ in range(self.rozum.num_joints)] + [1., ])
        self.action_space = gym.spaces.Box(low=low,
                                           high=high)
        angle_bounds = self.rozum.get_joint_intervals()[1]
        low = np.array([bound[0] for bound in angle_bounds] + [0., ])
        high = np.array([bound[0] + bound[1] for bound in angle_bounds] + [1., ])
        self.angles_bounds = gym.spaces.Box(low=low[:-1],
                                            high=high[:-1])
        self._available_obs_spaces = dict()
        self._render_dict = dict()
        self._available_obs_spaces['pov'] = gym.spaces.Box(shape=self.camera.resolution + [3],
                                                           low=0, high=255, dtype=np.uint8)
        self._render_dict['pov'] = self.get_image
        low = np.array([bound[0] for bound in angle_bounds] * 2 + [0., 0., -1., -1., -1.])
        high = np.array([bound[0] + bound[1] for bound in angle_bounds] * 2 + [1., 1., 1., 1., 1.])
        self._available_obs_spaces['arm'] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._render_dict['arm'] = self.get_arm_state
        self._available_obs_spaces['cube'] = gym.spaces.Box(shape=(6,),
                                                            low=-np.inf, high=np.inf,
                                                            dtype=np.float32)
        self._render_dict['cube'] = self.get_cube_state
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

        # Environment settings
        self.reward_range = None
        self.current_step = 0
        self.step_limit = 200
        self._start_arm_joint_pos = self.rozum.get_joint_positions()
        self._start_gripper_joint_pos = self.gripper.get_joint_positions()
        self.init_cube_pose = self.cube.get_pose()
        self.pose_sigma = pose_sigma

        # Video
        self.recording = list()
        self.current_episode = 0
        self.rewards = [0]
        self.path = video_path
        if video_path:
            def video_step():
                self._pyrep.step()
                self.recording.append(self.get_image()[..., ::-1])
            self.sim_step = video_step
        else:
            self.sim_step = self._pyrep.step

    def get_arm_state(self):
        arm = self.rozum.get_joint_positions()
        arm += self.rozum.get_joint_target_positions()
        arm += self.gripper.get_open_amount()
        arm += self.rozum_tip.get_position().tolist()
        return arm

    def get_cube_state(self):
        box = self.cube.get_position().tolist()
        box += self.cube.get_orientation().tolist()
        return box

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action: list):
        done = False
        info = dict()
        joint_action, ee_action = action[:-1], action[-1]
        distance_mod = 3
        scale = 100  # m -> cm

        previous_n = int(self._get_distance() * scale) // distance_mod
        grasped = self._robot_step(ee_action, joint_action)
        _, _, arm_z = self.rozum.joints[-1].get_position()
        tx, ty, tz = self.cube.get_position()
        pose_filter = arm_z > (tz + 0.05)
        current_distance = self._get_distance()
        current_n = int(current_distance * scale) // distance_mod
        reward = (previous_n - current_n) * pose_filter
        state = self.render()
        info['distance'] = current_distance
        if grasped:
            reward += 10
            done = True
            info['grasped'] = 1
        elif self.current_step >= self.step_limit:
            done = True
            info['grasped'] = 0
        self.rewards.append(reward)
        return state, reward, done, info

    def _get_distance(self):
        x, y, z = self.rozum_tip.get_position()
        tx, ty, tz = self.cube.get_position()
        distance = np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        return distance

    def _robot_step(self, ee_action, joint_action):
        grasped = False
        current_ee = (1.0 if np.mean(self.gripper.get_open_amount()) > 0.9
                      else 0.0)
        if ee_action > 0.5:
            ee_action = 1.0
        elif ee_action < 0.5:
            ee_action = 0.0
        if current_ee != ee_action:
            gripper_done = False
            self.rozum.set_joint_target_positions(self.rozum.get_joint_positions())
            while not gripper_done:
                gripper_done = self.gripper.actuate(ee_action, velocity=0.2)
                self.sim_step()
                self.current_step += 1
            if ee_action == 0.0:
                for g_obj in self.graspable_objects:
                    grasped = self.gripper.grasp(g_obj)
        else:
            position = [j + a * scale for j, a, scale in zip(self.rozum.get_joint_positions(),
                                                             joint_action, self.angles_scale)]
            position = list(np.clip(position, self.angles_bounds.low, self.angles_bounds.high))
            self.rozum.set_joint_target_positions(position)
            for _ in range(4):
                self.sim_step()
                self.current_step += 1
        return grasped

    def reset(self):
        self.gripper.release()
        arm, gripper = self._initial_robot_state
        self._pyrep.set_configuration_tree(arm)
        self._pyrep.set_configuration_tree(gripper)
        self.rozum.set_joint_positions(self._start_arm_joint_pos)
        self.rozum.set_joint_target_velocities(
            [0] * len(self.rozum.joints))
        self.gripper.set_joint_positions(
            self._start_gripper_joint_pos)
        self.gripper.set_joint_target_velocities(
            [0] * len(self.gripper.joints))

        # Initialize scene
        pose = self.init_cube_pose.copy()
        pose[0] += np.random.uniform(0.0, 0.2)  # max distance ~ 0.76
        pose[1] += np.random.uniform(-0.15, 0.15)
        self.cube.set_pose(pose)
        random_action = np.random.normal(0., self.pose_sigma, len(self._start_arm_joint_pos)) / 180 * np.pi
        position = [angle + action for angle, action in zip(self._start_arm_joint_pos, random_action)]
        self.rozum.set_joint_target_positions(position)
        for _ in range(4):
            self._pyrep.step()
        self.current_step = 0
        state = self.render()

        # Video
        if len(self.recording) > 0:
            name = str(self.current_episode).zfill(4) + "r" + str(sum(map(int, self.rewards))).zfill(4) + ".mp4"
            full_path = os.path.join(self.path, name)
            self.save_video(full_path, video=self.recording)
            self.current_episode += 1
            self.rewards = [0]
            self.recording = list()
        self.sim_step()
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

    @staticmethod
    def save_video(filename, video):
        """
        saves video from list of np.array images
        :param filename: filename or path to file
        :param video: [image, ..., image]
        :return:
        """
        size_x, size_y, size_z = video[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (size_x, size_y))
        for image in video:
            out.write(image)
        out.release()
        cv2.destroyAllWindows()
