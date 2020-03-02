from environments.env import RozumEnv
from utils.wrappers import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file and port parser")
    parser.add_argument('-port', action='store', type=int, default=19999, required=False)
    parser.add_argument('-robot_run_file', action='store', type=str, default='coppeliaSim.sh', required=False)
    params = parser.parse_args().__dict__

    env = RozumEnv(**params)
    env = FrameSkip(env)
    env = FrameStack(env)
    a = env.sample_action()
    env.step(a)
    a = env.reset()
    print(a.shape)
    env.close()