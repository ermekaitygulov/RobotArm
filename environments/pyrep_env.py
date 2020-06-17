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
                 headless=True, video_path='train/'):
        self.obs_space_keys = (obs_space_keys,) if isinstance(obs_space_keys, str) else obs_space_keys
        self._pyrep = PyRep()
        self._pyrep.launch(scene_file, headless=headless)
        self._pyrep.start()
        self.rozum = Rozum()
        self.rozum.set_control_loop_enabled(True)
        self.gripper = BaxterGripper()
        self.gripper.set_control_loop_enabled(True)
        self.cube = Shape("Cube")
        self.graspable_objects = [self.cube, ]
        self.camera = VisionSensor("render")
        self.rozum_tip = self.rozum.get_tip()

        self.angles_scale = np.array([np.pi for _ in range(self.rozum.num_joints)])
        low = np.array([-0.5 for _ in range(self.rozum.num_joints)] + [0., ])
        high = np.array([0.5 for _ in range(self.rozum.num_joints)] + [1., ])
        self.action_space = gym.spaces.Box(low=low,
                                           high=high)

        low = np.array([-0.8 * scale for scale in self.angles_scale] + [0., ])
        high = np.array([0.8 * scale for scale in self.angles_scale] + [1., ])
        self.angles_bounds = gym.spaces.Box(low=low[:-1],
                                            high=high[:-1])
        self._available_obs_spaces = dict()
        self._render_dict = dict()
        self._available_obs_spaces['pov'] = gym.spaces.Box(shape=self.camera.resolution + [3],
                                                           low=0, high=255, dtype=np.uint8)
        self._render_dict['pov'] = self.get_image
        low = np.array([-angle for angle in self.angles_scale] + [0., 0., -1., -1., -1.])
        high = np.array([angle for angle in self.angles_scale] + [1., 1., 1., 1., 1.])
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
        self.step_limit = 200
        self.init_angles = self.rozum.get_joint_positions()
        self.init_cube_pose = self.cube.get_pose()

        #Video
        self._eps_done = 0
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
        arm += self.gripper.get_open_amount()
        arm += self.rozum_tip.get_position().tolist()
        return arm

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action: list):
        done = False
        info = dict()
        joint_action, ee_action = action[:-1], action[-1]
        current_ee = (1.0 if np.mean(self.gripper.get_open_amount()) > 0.9
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
        x, y, z = self.rozum_tip.get_position()

        tx, ty, tz = self.cube.get_position()
        current_distance = np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        reward = tolerance(current_distance, (0.0, 0.01), 0.1)/20
        state = self.render()

        info['distance'] = current_distance
        if grasped:
            reward += 10
            done = True
            info['grasped'] = 1
        elif self.current_step >= self.step_limit:
            done = True
            info['grasped'] = 0
        if done:
            self._eps_done += 1
        self.rewards.append(reward)
        return state, reward, done, info

    def reset(self):
        self._pyrep.stop()
        self._pyrep.start()
        pose = self.init_cube_pose.copy()
        pose[0] += np.random.uniform(-0.05, 0.2)
        pose[1] += np.random.uniform(-0.3, 0.1)
        self.cube.set_pose(pose)
        state = self.render()
        self.current_step = 0

        #Video
        if len(self.recording) > 0:
            name = str(self.current_episode).zfill(4) + "r" + str(sum(map(int, self.rewards))).zfill(4) + ".mp4"
            full_path = os.path.join(self.path, name)
            self.save_video(full_path, video=self.recording)
            self.current_episode += 1
            self.rewards = [0]
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
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (size_x, size_y))
        for image in video:
            out.write(image)
        out.release()
        cv2.destroyAllWindows()
