import ray

from algorithms.apex import Learner, ParameterServer, Actor
from replay_buffers.apex_buffer import ApeXBuffer
from algorithms.model import ClassicCnn, DuelingModel
from environments.pyrep_env import RozumEnv
from utils.wrappers import *
from utils.util import config_gpu
import os


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ray.init(webui_host='127.0.0.1')
    n_actors = 3
    config_gpu()

    def make_env(name, test=False):
        env = RozumEnv()
        if test:
            SaveVideoWrapper(env)
            clear_mod = 10
        else:
            clear_mod= None
        env = FrameSkip(env)
        env = FrameStack(env, 2)
        env = AccuracyLogWrapper(env, 10, name, clear_mod)
        discrete_dict = dict()
        robot_dof = env.action_space.shape[0]
        for i in range(robot_dof):
            discrete_dict[i] = [5 if j == i else 0 for j in range(robot_dof)]
            discrete_dict[i + robot_dof] = [-5 if j == i else 0 for j in range(robot_dof)]
        env = DiscreteWrapper(env, discrete_dict)
        return env

    test_env = make_env()
    obs_shape = test_env.observation_space.shape
    action_shape = test_env.action_space.n
    test_env.close()

    def make_model(name, input_shape, output_shape):
        import tensorflow as tf
        base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])
        head = DuelingModel([1024], output_shape)
        model = tf.keras.Sequential([base, head], name)
        model.build((None, ) + input_shape)
        return model

    parameter_server = ParameterServer.remote()
    replay_buffer = ApeXBuffer.remote(int(1e5))
    learner = Learner.remote(replay_buffer, make_model, obs_shape, action_shape,
                             parameter_server, update_target_net_mod=1000, gamma=0.99, learning_rate=1e-4,
                             batch_size=32, replay_start_size=1000)
    actors = [Actor.remote(i, replay_buffer,  make_model, obs_shape, action_shape,
                           make_env, parameter_server, gamma=0.99, n_step=10, sync_nn_mod=100, send_rollout_mod=64,
                           test=(i == (n_actors-1))) for i in range(n_actors)]

    processes = list()
    processes.append(learner.update.remote(max_eps=1e+6, log_freq=10))
    for i, a in enumerate(actors[:-1]):
        processes.append(a.train.remote(epsilon=0.1, final_epsilon=0.01, eps_decay=0.99,
                                        max_eps=1e+6, send_rollout_mod=64, sync_nn_mod=100))
    processes.append(actors[-1].validate.remote(test_mod=100, test_eps=10))
    ray.wait(processes)
    ray.timeline()
