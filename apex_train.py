import ray

from algorithms.apex import Learner, ParameterServer, Actor
from replay_buffers.apex_buffer import PrioritizedBuffer
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

    n_actors = 3

    env = [RozumEnv() for _ in range(n_actors)]
    env[-1] = SaveVideoWrapper(env[-1])
    env = [FrameSkip(e) for e in env]
    env = [FrameStack(e, 2) for e in env]
    env = [AccuracyLogWrapper(e, 10) for e in env[:-1]]
    discrete_dict = dict()
    robot_dof = env[0].action_space.shape[0]
    for i in range(robot_dof):
        discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
        discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
    env = [DiscreteWrapper(e, discrete_dict) for e in env]

    def make_model(name):
        base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])
        head = DuelingModel([1024], env[0].action_space.n)
        model = tf.keras.Sequential([base, head], name)
        model.build((None, ) + env[0].observation_space.shape)
        return model
    config = None
    parameter_server = ParameterServer.remote(config)
    replay_buffer = PrioritizedBuffer.remote(int(1e5))
    learner = Learner.remote(replay_buffer, parameter_server, make_model)
    actors = [Actor.remote(i, 0.99, 10, replay_buffer, make_model, parameter_server) for i in range(n_actors)]
    summary_writer = tf.summary.create_file_writer('train/')
    with summary_writer.as_default():
        processes = list()
        processes.append(learner.update.remote())
        for i, a in enumerate(actors[:-1]):
            processes.append(a.train.remote(env[i]))
        processes.append(actors[-1].validate.remote(env[-1]))
    ray.wait(processes)
    ray.timeline()
    for e in env:
        e.close()
