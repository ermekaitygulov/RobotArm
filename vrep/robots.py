import numpy as np
import subprocess
import signal
import math
import os
import sys
import time
import cv2
import vrep.vrep as vrep
import vrep.vrepConst as Const_v


class Handler:
    def __init__(self, handle, get_func, set_func=None):
        self._handle = handle
        self._get_func = get_func
        self._set_func = set_func

    @property
    def value(self):
        return self._get_func(self._handle)

    @value.setter
    def value(self, value):
        if self._set_func:
            self._set_func(self._handle, value)


class Rozum:
    def __init__(self, run_file='coppeliaSim.sh', port=19999):

        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if b'vrep' in line:
                pid = int(line.split(None, -1)[0])
                os.kill(pid, signal.SIGKILL)

        os.chdir(os.environ['VREP_PATH'])
        os.system("./{} -h -s {} &".format(run_file, os.environ['ROZUM_MODEL_PATH']))
        os.chdir('..')
        vrep.simxFinish(-1)
        time.sleep(1)

        self.DoF = 6
        self.ID = vrep.simxStart('127.0.0.1', port, True, False, 5000, 5)
        self.opM_get = Const_v.simx_opmode_blocking
        self.opM_set = Const_v.simx_opmode_oneshot
        # check the connection
        if self.ID != -1:
            print("Connected")
        else:
            sys.exit("Error")

        self.gripper_motor = self.get_handle('RG2_openCloseJoint')
        self.action_bound = [-180, 180]

        self.joints = [Handler(self.get_handle("joint{}".format(i)), self.get_joint_angle, self.set_joint_angle)
                       for i in range(self.DoF)]
        self.side_cam = Handler(self.get_handle('Vision_sensor'), self.get_image)
        self.side_cam_dim = self.side_cam.value.shape
        self.on_arm_cam = Handler(self.get_handle('render'), self.get_image)
        self.tip = Handler(self.get_handle("Tip"), self.get_position)
        self.cube = Handler(self.get_handle("Cube"), self.get_position, self.set_position)
        self.goal = Handler(self.get_handle("Goal"), self.get_position, self.set_position)

        self.str_simx_return = [
            'simx_return_ok',
            'simx_return_novalue_flag',
            'simx_return_timeout_flag',
            'simx_return_illegal_opmode_flag',
            'simx_return_remote_error_flag',
            'simx_return_split_progress_flag',
            'simx_return_local_error_flag',
            'simx_return_initialize_error_flag']

    def get_handle(self, name):
        (check, handle) = vrep.simxGetObjectHandle(self.ID, name, self.opM_get)
        if check != 0:
            print("Couldn't find %s" % name)
        return handle

    def get_position(self, handle):
        _, pose = vrep.simxGetObjectPosition(self.ID, handle, -1, self.opM_get)
        return np.array(pose)

    def set_position(self, handle, pose):
        vrep.simxSetObjectPosition(self.ID, handle, -1, pose, self.opM_set)

    def get_orientation(self, handle):
        (code, ornt) = vrep.simxGetObjectOrientation(self.ID, handle, -1, self.opM_get)
        return np.array([np.sin(ornt[0]), np.cos(ornt[0]),
                         np.sin(ornt[1]), np.cos(ornt[1]),
                         np.sin(ornt[2]), np.cos(ornt[2])])

    def close_gripper(self):
        _ = vrep.simxSetJointForce(self.ID, self.gripper_motor, 20, self.opM_get)
        _ = vrep.simxSetJointTargetVelocity(self.ID, self.gripper_motor, -0.05, self.opM_get)

    def open_gripper(self):
        _ = vrep.simxSetJointForce(self.ID, self.gripper_motor, 20, self.opM_get)
        _ = vrep.simxSetJointTargetVelocity(self.ID, self.gripper_motor, 0.05, self.opM_get)

    def get_image(self, handle):
        res, im = self.rapi_rc(vrep.simxGetVisionSensorImage(self.ID, handle, 0, self.opM_get))
        img = np.array(im, dtype=np.uint8)
        img.resize([res[0], res[1], 3])
        img = cv2.flip(img, 0)
        # img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        return img

    def set_joint_angle(self, handle, value):
        # in radian
        clipped_value = min(max(value, self.action_bound[0]), self.action_bound[1])
        vrep.simxSetJointTargetPosition(self.ID, handle, clipped_value * math.pi / 180,
                                        self.opM_set)

    def get_joint_angle(self, handle):
        _, angle = vrep.simxGetJointPosition(self.ID, handle, self.opM_get)
        angle *= 180 / math.pi
        return angle

    def stop_simulation(self):
        self.rapi_rc(vrep.simxStopSimulation(self.ID, self.opM_get))

        # Checking if the server really stopped
        try:
            while True:
                self.rapi_rc(vrep.simxGetIntegerSignal(self.ID, 'sig_debug', self.opM_get))
                e = vrep.simxGetInMessageInfo(self.ID, Const_v.simx_headeroffset_server_state)
                still_running = e[1] & 1
                if not still_running:
                    break
        except:
            pass

    def disconnect(self):
        # Clearing debug signal
        vrep.simxClearIntegerSignal(self.ID, 'sig_debug', self.opM_get)
        vrep.simxFinish(self.ID)

    def rapi_rc(self, ret_tuple, tolerance=Const_v.simx_return_novalue_flag):
        istuple = isinstance(ret_tuple, tuple)
        ret = ret_tuple[0] if istuple else ret_tuple
        if (ret != Const_v.simx_return_ok) and (ret != tolerance):
            raise RuntimeError(
                'Remote API return code: (' + str(ret) + ': ' + self.str_simx_return[ret.bit_length()] + ')')

        return ret_tuple[1:] if istuple else None
