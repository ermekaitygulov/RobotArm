from agent.algorithms import DQN
from agent.replay_buffers import PrioritizedBuffer
from agent.model import classic_cnn, DuelingModel
from environments.env import RozumEnv
from utils.wrappers import *
import argparse
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file and port parser")
    parser.add_argument('-port', action='store', type=int, default=19999, required=False)
    parser.add_argument('-robot_run_file', action='store', type=str, default='coppeliaSim.sh', required=False)
    params = parser.parse_args().__dict__

    env = RozumEnv(**params)
    env = FrameSkip(env)
    env = FrameStack(env, 2)
    replay_buffer = PrioritizedBuffer(1e6)
    def make_model():
        conv_base = classic_cnn([8, 16, 32], [3, 3, 3], [2, 2, 2])
        return DuelingModel(conv_base, [1024, 512], env.action_dim)
    agent = DQN(env.action_dim, env.obs_dim, replay_buffer, make_model(), make_model())
    summary_writer = tf.summary.create_file_writer('/train')
    with summary_writer.as_default():
        agent.train(env)
    env.close()