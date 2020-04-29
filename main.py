from algorithms.dqn.run import dqn_run
from algorithms.apex.run import apex_run
from argparse import ArgumentParser
import os

algorithms = {'dqn': dqn_run,
              'apex': apex_run}

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = ArgumentParser()
    available_algorithms = ', '.join(algorithms.keys())
    parser.add_argument('--alg', action='store', help='Available', type=str, required=True)
