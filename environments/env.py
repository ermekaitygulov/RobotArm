import vrep.vrep as vrep
import vrep.vrepConst as const_v
import time
import sys
import numpy as np
import cv2
import math
import os
import gym
import subprocess, signal
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from threading import Thread

class Vrep:
    def __init__(self):

        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if b'vrep' in line:
                pid = int(line.split(None, -1)[0])
                os.kill(pid, signal.SIGKILL)

        self.vrep_root = os.getenv('VREP_PATH')
        self.scene_file = os.getenv('ROZUM_MODEL_PATH')
        os.chdir(self.vrep_root)
        os.system("./coppeliaSim.sh -s " + self.scene_file + " &")

        vrep.simxFinish(-1)
        time.sleep(1)

        self.DoF = 6
        self.ID = vrep.simxStart('127.0.0.1', 19999, True, False, 5000, 5)
        # check the connection
        if self.ID != -1:
            print("Connected")
        else:
            sys.exit("Error")

        self.gripper_motor = self.get_handle('RG2_openCloseJoint')
        self.action_bound = [-180, 180]

        self.joints = [self.create_property(self.get_handle(f"joint{i}"), self.get_joint_angle, self.set_joint_angle)
                       for i in range(self.DoF)]
        self.cam_handle = self.get_handle('Vision_sensor')
        self.render_handle = self.get_handle('render')

        self.tip = self.create_property(self.get_handle("Tip"), self.get_position)
        self.cube = self.create_property(self.get_handle("Cube"), self.get_position, self.set_position)
        self.goal = self.create_property(self.get_handle("Goal"), self.get_position, self.set_position)

    @staticmethod
    def create_property(handle, get_function=None, set_function=None):
        @property
        def f():
            return get_function(handle)

        @f.setter
        def f(value):
            return set_function(handle, value)

        return f

    def get_handle(self, name):
        (check, handle) = vrep.simxGetObjectHandle(self.ID, name, const_v.simx_opmode_blocking)
        if check != 0:
            print("Couldn't find %s" % name)
        return handle

    def get_position(self, handle):
        _, pose = vrep.simxGetObjectPosition(self.ID, handle, -1, const_v.simx_opmode_buffer)
        return np.array(pose)

    def set_position(self, handle, pose):
        vrep.simxSetObjectPosition(self.ID, handle, -1, pose, const_v.simx_opmode_oneshot_wait)

    def get_orientation(self,handle):
        (code, ornt) = vrep.simxGetObjectOrientation(self.ID, handle, -1, const_v.simx_opmode_buffer)
        return np.array([np.sin(ornt[0]), np.cos(ornt[0]),
                         np.sin(ornt[1]), np.cos(ornt[1]),
                         np.sin(ornt[2]), np.cos(ornt[2])])

    def close_gripper(self, render=False):
        _ = vrep.simxSetJointForce(self.ID, self.gripper_motor, 20, const_v.simx_opmode_blocking)
        _ = vrep.simxSetJointTargetVelocity(self.ID, self.gripper_motor, -0.05, const_v.simx_opmode_blocking)

    def open_gripper(self, render=False):
        _ = vrep.simxSetJointForce(self.ID, self.gripper_motor, 20, const_v.simx_opmode_blocking)
        _ = vrep.simxSetJointTargetVelocity(self.ID, self.gripper_motor, 0.05, const_v.simx_opmode_blocking)

    def get_image(self):
        _, res, im = vrep.simxGetVisionSensorImage(self.ID, self.cam_handle, 0, const_v.simx_opmode_buffer)
        img1 = np.array(im, dtype=np.uint8)
        img1.resize([res[0], res[1], 3])
        img1 = cv2.flip(img1, 0)
        _, res, im = vrep.simxGetVisionSensorImage(self.ID, self.render_handle, 0, const_v.simx_opmode_buffer)
        img2 = np.array(im, dtype=np.uint8)
        img2.resize([res[0], res[1], 3])
        img2 = cv2.flip(img2, 0)
        return img1, img2

    def set_joint_angle(self, handle, value):
        # in radian
        clipped_value = min(max(value, self.action_bound[0]), self.action_bound[1])
        vrep.simxSetJointTargetPosition(self.ID, handle, clipped_value * math.pi / 180,
                                        const_v.simx_opmode_blocking)

    def get_joint_angle(self, handle):
        _, angle = vrep.simxGetJointPosition(self.ID, handle, const_v.simx_opmode_buffer)
        angle *= 180 / math.pi


class RozumEnv:

    def __init__(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (1024, 1024))
        self.vrep = Vrep()
        # self.action_bound = [[-15,15],[-10,110],[-30,30],[-120,120],[-180,180],[-180,180]]
        self.action_range = [-5, 5]
        self.action_dim = self.vrep.DoF

        self.action_space = gym.spaces.Box(shape=(self.action_dim,), low=-5, high=5)
        self.observation_space = gym.spaces.Box(shape=(3 + self.action_dim * 2,), low=-180, high=180)
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]

        self.init_angles = self.vrep.joints
        self.init_cube_pose = self.vrep.cube
        self.init_goal_pose = self.vrep.goal

    def sample_action(self):
        return np.random.uniform(*self.vrep.action_bound, size=self.action_dim)

    def step(self, action):
        for i in range(self.action_dim):
            self.vrep.joints[i] = self.vrep.joints[i] + action[i]
        time.sleep(0.3)
        pose = self.vrep.tip
        r = 0.0
        done = False
        target = self.vrep.cube
        s = self.vrep.get_image()
        d = np.linalg.norm(pose - target)
        r += (-d - 0.01 * np.square(action).sum())
        if d < 0.02:
            done = True
        return s, r, done, None

    def reset(self):
        for i in range(self.action_dim):
            self.vrep.joints[i] = self.init_angles[i]
        self.vrep.open_gripper()
        self.vrep.cube = self.init_cube_pose
        self.vrep.goal = self.init_goal_pose
        time.sleep(2)
        s, _ = self.vrep.get_image()
        return s

    def render(self):
        img = self.vrep.get_image()
        self.out.write(img)


if __name__ == '__main__':
    env = RozumEnv()
    a = env.sample_action()
    print(env.vrep.joints[0])
    print(env.vrep.cube)
    env.reset()
