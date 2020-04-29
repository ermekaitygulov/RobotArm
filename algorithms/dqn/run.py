from algorithms.dqn.dqn import DQN
from common.cpprb_wrapper import PER
from algorithms.model import ClassicCnn, DuelingModel, MLP
from environments.pyrep_env import RozumEnv
from common.wrappers import *
import tensorflow as tf
import os
from common.util import config_gpu


def dqn_run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tf.config.optimizer.set_jit(True)
    config_gpu()

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
    env_dict = {'action': {'dtype': 'int32'},
                'reward': {'dtype': 'float32'},
                'done': {'dtype': 'bool'},
                'n_reward': {'dtype': 'float32'},
                'n_done': {'dtype': 'bool'},
                'actual_n': {'dtype': 'float32'}
                }
    for prefix in ('', 'n_'):
        env_dict[prefix + 'pov'] = {'shape': env.observation_space['pov'].shape,
                                    'dtype': 'uint8'}
        env_dict[prefix + 'angles'] = {'shape': env.observation_space['angles'].shape,
                                       'dtype': 'float32'}

    replay_buffer = PER(size=50000, state_prefix=('', 'next_', 'n_'),
                        state_keys=('pov', 'angles'), env_dict=env_dict, next_of=('pov', 'angles'))

    def make_model(name, obs_space, action_space):
        pov = tf.keras.Input(shape=obs_space['pov'].shape)
        angles = tf.keras.Input(shape=obs_space['angles'].shape)
        normalized_pov = pov / 255
        pov_base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])(normalized_pov)
        angles_base = MLP([512, 256])(angles)
        base = tf.keras.layers.concatenate([pov_base, angles_base])
        head = DuelingModel([1024], action_space.n)(base)
        model = tf.keras.Model(inputs={'pov': pov, 'angles': angles}, outputs=head, name=name)
        return model

    agent = DQN(replay_buffer, make_model, env.observation_space, env.action_space, replay_start_size=100,
                train_quantity=100, train_freq=100, log_freq=20)
    summary_writer = tf.summary.create_file_writer('train/')
    with summary_writer.as_default():
        agent.train(env, 1000)
    env.close()


if __name__ == '__main__':
    dqn_run()