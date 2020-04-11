from algorithms.dqn import DQN
from replay_buffers.replay_buffers import PrioritizedReplayBuffer
from algorithms.model import ClassicCnn, DuelingModel, MLP
from environments.pyrep_env import RozumEnv
from utils.wrappers import *
import tensorflow as tf
import os


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    tf.debugging.set_log_device_placement(False)
    tf.config.optimizer.set_jit(True)

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
    env = FrameStack(env, 2, stack_key='pov')
    env = AccuracyLogWrapper(env, 10)
    discrete_dict = dict()
    robot_dof = env.action_space.shape[0]
    for i in range(robot_dof):
        discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
        discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
    env = DiscreteWrapper(env, discrete_dict)
    replay_buffer = PrioritizedReplayBuffer(50000)

    def make_model(name, obs_space, action_space):
        pov = tf.keras.Input(shape=obs_space['pov'].shape)
        angles = tf.keras.Input(shape=obs_space['angles'].shape)
        pov_base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])(pov)
        angles_base = MLP([512, 256])(angles)
        base = tf.keras.layers.concatenate([pov_base, angles_base])
        head = DuelingModel([1024], action_space.n)(base)
        model = tf.keras.Model(inputs={'pov': pov, 'angles': angles}, outputs=head, name=name)
        return model
    agent = DQN(replay_buffer, make_model, env.observation_space, env.action_space, replay_start_size=100)
    summary_writer = tf.summary.create_file_writer('train/')
    with summary_writer.as_default():
        agent.train(env, 1000)
    env.close()
