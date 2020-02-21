import numpy as np
import subprocess, signal
import math
import os
import sys
import time
import cv2
import vrep.vrep as vrep
import vrep.vrepConst as const_v


class Handler:
    def __init__(self, handle, get_func, set_func=None):
        self._handle = handle
        self._get_func = get_func
        self._set_func = set_func
        self.value

    @property
    def value(self):
        return self._get_func(self._handle)

    @value.setter
    def value(self, value):
        if self._set_func:
            return self._set_func(self._handle, value)

class Rozum:
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

        self.joints = [Handler(self.get_handle("joint{}".format(i)), self.get_joint_angle, self.set_joint_angle)
                       for i in range(self.DoF)]
        self.cam_handle = self.get_handle('Vision_sensor')
        self.render_handle = self.get_handle('render')

        self.tip = Handler(self.get_handle("Tip"), self.get_position)
        self.cube = Handler(self.get_handle("Cube"), self.get_position, self.set_position)
        self.goal = Handler(self.get_handle("Goal"), self.get_position, self.set_position)

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
        print(img1.shape)
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
        return angle