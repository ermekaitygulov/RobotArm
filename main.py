from agent.algorithms import DQN
from agent.replay_buffers import PrioritizedBuffer
from agent.model import DuelingModel
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
    with tf.InteractiveSession() as sess:
        replay_buffer = PrioritizedBuffer(1e6)
        conv_base = None
        def make_model(name):
            return DuelingModel(conv_base, [1024, 512], env.action_dim, env.obs_dim, sess, name)
        config = 123
        agent = DQN(config, env.action_dim, env.obs_dim, replay_buffer, make_model('Q_network'), make_model('Q_target'),
                    sess)
        agent.train(env)
        env.close()