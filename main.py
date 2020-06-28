
from algorithms.apex.run import apex_run
from argparse import ArgumentParser
import os

algorithms = {'apex': apex_run}

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
