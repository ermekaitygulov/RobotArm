from algorithms.dqn import DQN
from replay_buffers.replay_buffers import PrioritizedBuffer
from algorithms.model import ClassicCnn, DuelingModel
from environments.pyrep_env import RozumEnv
from utils.wrappers import *
import tensorflow as tf
import os


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    tf.debugging.set_log_device_placement(False)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    env = RozumEnv()
    # env = SaveVideoWrapper(env)
    env = FrameSkip(env)
    env = FrameStack(env, 2)
    env = AccuracyLogWrapper(env, 10)
    discrete_dict = dict()
    robot_dof = env.action_space.shape[0]
    for i in range(robot_dof):
        discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
        discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
    env = DiscreteWrapper(env, discrete_dict)
    replay_buffer = PrioritizedBuffer(int(1e5))

    def make_model(name):
        base = ClassicCnn([16, 32, 64], [3, 3, 3], [2, 2, 2])
        head = DuelingModel([1024], env.action_space.n)
        model = tf.keras.Sequential([base, head], name)
        model.build((None, ) + env.observation_space.shape)
        return model
    agent = DQN(env.action_space.n, replay_buffer, make_model('Online_model'), make_model('Target_model'))
    summary_writer = tf.summary.create_file_writer('train/')
    with summary_writer.as_default():
        agent.train(env, 1000)
    env.close()