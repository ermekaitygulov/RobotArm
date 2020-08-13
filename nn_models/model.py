import gym
from common.tf_util import concatenate
from nn_models.building_blocks import *
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def get_network_builder(name, *args, **kwargs):
    """
    If you want to register your own network outside model.py, you just need:
    Usage Example:
    -------------
    from algorithms.model import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn
    """
    if callable(name):
        return name
    elif name in mapping:
        func = mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))

    def thunk(model_name, obs_space, act_space):
        return func(model_name, obs_space, act_space, *args, **kwargs)
    return thunk


@register("DuelingDQN_pov_arm")
def make_model(name, obs_space, action_space, reg=1e-6):
    pov = tf.keras.Input(shape=obs_space['pov'].shape)
    arm = tf.keras.Input(shape=obs_space['arm'].shape)
    normalized_pov = pov / 255
    pov_base = make_cnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2], 'tanh', reg)(normalized_pov)
    angles_base = make_mlp([512, 256], 'tanh', reg)(arm)
    base = concatenate([pov_base, angles_base])
    head = DuelingModel([1024], action_space.n, reg)(base)
    model = tf.keras.Model(inputs={'pov': pov, 'arm': arm}, outputs=head, name=name)
    return model


def make_uni_base(img, feat, reg):
    bases = list()
    if len(img) > 0:
        for i in img.values():
            normalized = i/255
            cnn = make_impala_cnn((16, 32, 32), reg, flat=True, use_bn=False)(normalized)
            bases.append(cnn)
    if len(feat) > 0:
        feat_base = concatenate(list(feat.values()))
        mlp = make_mlp([64, 64], 'relu', reg)(feat_base)
        bases.append(mlp)
    base = concatenate(bases)
    return base


@register("DuelingDQN_uni")
def make_model(name, obs_space, action_space, reg=1e-6):
    img = dict()
    feat = dict()
    if isinstance(obs_space, gym.spaces.Dict):
        for key, value in obs_space.spaces.items():
            if len(value.shape) > 1:
                img[key] = tf.keras.Input(shape=value.shape)
            else:
                feat[key] = tf.keras.Input(shape=value.shape)
    else:
        if len(obs_space.shape) > 1:
            img['state'] = tf.keras.Input(shape=obs_space.shape)
        else:
            feat['state'] = tf.keras.Input(shape=obs_space.shape)
    base = make_uni_base(img, feat, reg)
    head = DuelingModel([1024, 512], action_space.n, reg)(base)
    model = tf.keras.Model(inputs={**img, **feat}, outputs=head, name=name)
    return model


@register("DuelingDQN_arm_cube")
def make_model(name, obs_space, action_space, reg=1e-7, noisy_head=False):
    cube = tf.keras.Input(shape=obs_space['cube'].shape)
    arm = tf.keras.Input(shape=obs_space['arm'].shape)
    features = concatenate([arm, cube])
    base = make_mlp([400, 300], 'tanh', reg)(features)
    head = DuelingModel([512], action_space.n, reg, noisy=noisy_head)(base)
    model = tf.keras.Model(inputs={'cube': cube, 'arm': arm}, outputs=head, name=name)
    return model


@register("Critic_uni")
def make_critic(name, obs_space, action_space, reg=1e-6, noisy_head=False):
    img = dict()
    feat = dict()
    layer = NoisyDense if noisy_head else Dense
    if isinstance(obs_space, gym.spaces.Dict):
        for key, value in obs_space.spaces.items():
            if len(value.shape) > 1:
                img[key] = tf.keras.Input(shape=value.shape)
            else:
                feat[key] = tf.keras.Input(shape=value.shape)
    else:
        if len(obs_space.shape) > 1:
            img['state'] = tf.keras.Input(shape=obs_space.shape)
        else:
            feat['state'] = tf.keras.Input(shape=obs_space.shape)
    action = tf.keras.Input(shape=action_space.shape)
    feat['action'] = action
    base = make_uni_base(img, feat, reg)
    base = layer(256, 'relu', use_bias=True,  kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    base = layer(256, 'relu', use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    head = layer(1, use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    model = tf.keras.Model(inputs={**img, **feat}, outputs=head, name=name)
    return model


@register("Actor_uni")
def make_model(name, obs_space, action_space, reg=1e-6, noisy_head=False):
    img = dict()
    feat = dict()
    layer = NoisyDense if noisy_head else Dense
    if isinstance(obs_space, gym.spaces.Dict):
        for key, value in obs_space.spaces.items():
            if len(value.shape) > 1:
                img[key] = tf.keras.Input(shape=value.shape)
            else:
                feat[key] = tf.keras.Input(shape=value.shape)
    else:
        if len(obs_space.shape) > 1:
            img['state'] = tf.keras.Input(shape=obs_space.shape)
        else:
            feat['state'] = tf.keras.Input(shape=obs_space.shape)
    base = make_uni_base(img, feat, reg)
    base = layer(256, 'relu', use_bias=True,  kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    base = layer(256, 'relu', use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    head = layer(action_space.shape[0], use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    model = tf.keras.Model(inputs={**img, **feat}, outputs=head, name=name)
    return model
