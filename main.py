from algorithms.dqn.run import dqn_run
from algorithms.apex.run import apex_dqn_run, apex_ddpg_run
from algorithms.ddpg.run import ddpg_run
from algorithms.td3.run import td3_run
from argparse import ArgumentParser
import os

algorithms = {'dqn': dqn_run,
              'apex-dqn': apex_dqn_run,
              'ddpg': ddpg_run,
              'apex-ddpg': apex_ddpg_run,
              'td3': td3_run}

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = ArgumentParser()
    available_algorithms = ', '.join(algorithms.keys())
    parser.add_argument('--alg', action='store', help='Available algorithms: {}'.format(available_algorithms),
                        type=str, required=True)
    parser.add_argument('--config_path', action='store', help='Path to config with params for chosen alg',
                        type=str, required=True)
    args = parser.parse_args()
    algorithm_run = algorithms[args.alg]
    algorithm_run(args.config_path)

