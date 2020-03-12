from environments.pyrep_env import RozumEnv
import timeit
from utils.wrappers import DiscreteWrapper, SaveVideoWrapper, FrameSkip

env = RozumEnv()
env = FrameSkip(env)
done = False
start_time = timeit.default_timer()
discrete_dict = dict()
robot_dof = env.action_space.shape[0]
for i in range(robot_dof):
    discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
    discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
env = DiscreteWrapper(env, discrete_dict)

for i in range(75):
    action = env.sample_action()
    state, reward, done, _ = env.step(action)
    print("State shape: ", state.shape)
    print("Reward: ", reward)
    if i % 100 == 0:
        env.reset()
env.reset()

stop_time = timeit.default_timer()
print("RunTime: ", stop_time - start_time)
env.close()



