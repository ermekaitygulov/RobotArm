from argparse import ArgumentParser

import ray
import yaml

import algorithms
from algorithms.apex import Actor

from offpolicy_train import stack_env, make_discrete_env, make_continuous_env
from nn_models.model import get_network_builder
from environments.pyrep_env import RozumEnv
from common.wrappers import *
from tqdm import tqdm


def make_env(thread_id, n_actors, env_kwargs=None, frame_stack=4, discretize=True, save_path='train/'):
    env_kwargs = env_kwargs if env_kwargs else {}
    environment = RozumEnv(**env_kwargs)
    if frame_stack > 1:
        environment = stack_env(environment, frame_stack)
    if discretize:
        environment = make_discrete_env(environment, 0., 0.)
    else:
        environment = make_continuous_env(environment, 0., (0., 0.))
    environment = DataSave(environment, save_path, n_actors, thread_id)
    return environment


def make_remote_base(remote_config, n_actors):
    base = getattr(algorithms, remote_config['base'])

    def make_env_thunk(index):
        def thunk():
            return make_env(index, n_actors, **remote_config['env'])
        return thunk

    test_env = make_env_thunk(-1)()
    obs_space = test_env.observation_space
    action_space = test_env.action_space
    test_env.close()

    network_kwargs = dict()
    for arg_name, arg_value in remote_config['neural_network'].items():
        if isinstance(arg_value, dict):
            network_kwargs[arg_name] = get_network_builder(**arg_value)
        else:
            network_kwargs[arg_name] = get_network_builder(arg_value)

    if 'actor_resource' in remote_config:
        remote_actors = [Actor.options(**remote_config['actor_resource']).remote(thread_id=actor_id, base=base,
                                                                                 make_env=make_env_thunk(actor_id),
                                                                                 obs_space=obs_space,
                                                                                 action_space=action_space,
                                                                                 **network_kwargs,
                                                                                 **remote_config['actors'])
                         for actor_id in range(n_actors)]
    else:
        remote_actors = [Actor.remote(thread_id=actor_id, base=base, make_env=make_env_thunk(actor_id),
                                      obs_space=obs_space, action_space=action_space,
                                      **network_kwargs, **remote_config['actors']) for actor_id in range(n_actors)]
    return remote_actors


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', action='store', help='Path to config with params for chosen alg',
                        type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = defaultdict(dict)
        config.update(yaml.load(config_file, Loader=yaml.FullLoader))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
    os.environ["QT_DEBUG_PLUGINS"] = "0"
    ray.init(webui_host='0.0.0.0', num_gpus=1, log_to_driver=False)

    # Preparation
    generation_config = dict(max_eps=10, num_actors=4)
    if 'generate' in config.keys():
        for key, value in config['generate'].items():
            assert key in generation_config.keys()
            generation_config[key] = value
    actors = make_remote_base(config, generation_config['num_actors'])

    # Start tasks
    generators = {}
    for a in actors:
        generators[a.test.remote()] = a

    # Generation process
    for _ in tqdm(range(generation_config['max_eps'])):
        ready_ids, _ = ray.wait(list(generators))
        first_id = ready_ids[0]
        first = generators.pop(first_id)
        generators[first.test.remote()] = first
    ray.timeline()
