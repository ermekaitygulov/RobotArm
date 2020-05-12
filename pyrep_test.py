from environments.pyrep_env import RozumEnv
import timeit
from common.wrappers import DiscreteWrapper, FrameSkip, SaveVideoWrapper
from tqdm import tqdm
import numpy as np

env = RozumEnv(obs_space_keys=('arm', 'cube'))
env = FrameSkip(env)
done = False
start_time = timeit.default_timer()
discrete_dict = dict()
robot_dof = env.action_space.shape[0] - 1
for i in range(robot_dof):
    # joint actions
    discrete_angle = 5 / 180
    discrete_dict[i] = [discrete_angle / env.angles_scale[j]
                        if j == i else 0 for j in range(robot_dof)] + [1., ]
    discrete_dict[i + robot_dof] = [-discrete_angle / env.angles_scale[j]
                                    if j == i else 0 for j in range(robot_dof)] + [1., ]
# gripper action
discrete_dict[2 * robot_dof] = [0., ] * (robot_dof + 1)
env = DiscreteWrapper(env, discrete_dict)

for i in tqdm(range(400)):
    state, reward, done, _ = env.step(i % robot_dof)
    if i % 50 == 0:
        env.reset()
env.reset()

stop_time = timeit.default_timer()
print("RunTime: ", stop_time - start_time)
env.close()



