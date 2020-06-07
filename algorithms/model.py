import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow.keras.backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
import gym
from common.tf_util import concatenate

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def get_network_builder(name):
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
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))


class DuelingModel(tf.keras.Model):
    def __init__(self, units, action_dim, reg=1e-6, noisy=False):
        super(DuelingModel, self).__init__()
        reg = l2(reg)
        layer = NoisyDense if noisy else Dense
        self.h_layers = Sequential([layer(unit, 'relu', kernel_regularizer=reg, bias_regularizer=reg)
                                    for unit in units[:-1]])
        self.a_head = layer(units[-1]/2, 'relu', kernel_regularizer=reg, bias_regularizer=reg)
        self.v_head = layer(units[-1]/2, 'relu', kernel_regularizer=reg, bias_regularizer=reg)
        self.a_head1 = layer(action_dim, kernel_regularizer=reg, bias_regularizer=reg)
        self.v_head1 = layer(1, kernel_regularizer=reg, bias_regularizer=reg)

    @tf.function
    def call(self, inputs):
        print('Building model')
        features = self.h_layers(inputs)
        advantage, value = self.a_head(features), self.v_head(features)
        advantage, value = self.a_head1(advantage), self.v_head1(value)
        advantage = advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
        out = value + advantage
        return out


class NoisyDense(Dense):

    # factorized noise
    def __init__(self, units, *args, **kwargs):
        self.output_dim = units
        self.f = lambda x: tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        super(NoisyDense, self).__init__(units, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=None)

        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer=self.kernel_initializer,
                                            name='sigma_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=None)

        if self.use_bias:
            self.bias = self.add_weight(shape=(1, self.units),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=None)

            self.bias_sigma = self.add_weight(shape=(1, self.units,),
                                              initializer=self.bias_initializer,
                                              name='bias_sigma',
                                              regularizer=self.bias_regularizer,
                                              constraint=None)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        kernel_input = self.f(K.random_normal(shape=(self.input_dim, 1)))
        kernel_output = self.f(K.random_normal(shape=(1, self.units)))
        kernel_epsilon = tf.matmul(kernel_input, kernel_output)

        w = self.kernel + self.kernel_sigma * kernel_epsilon
        output = K.dot(inputs, w)

        if self.use_bias:
            b = self.bias + self.bias_sigma * kernel_output
            output = output + b
        if self.activation is not None:
            output = self.activation(output)
        return output


class TwinCritic(tf.keras.Model):
    def __init__(self, build_function, name, obs_space, action_space, reg=1e-6, noisy_head=True):
        super(TwinCritic, self).__init__()
        self.twin1 = build_function(name + '_1', obs_space, action_space, reg, noisy_head)
        self.twin2 = build_function(name + '_2', obs_space, action_space, reg, noisy_head)

    def call(self, inputs, **kwargs):
        return self.twin1(inputs), self.twin2(inputs)


def make_twin(build_critic):
    def thunk(name, obs_space, action_space, reg=1e-6, noisy_head=True):
        return TwinCritic(build_critic, name, obs_space, action_space, reg, noisy_head)
    return thunk


def make_mlp(units, activation='tanh', reg=1e-6, noisy=False):
    _reg = l2(reg)
    layer = NoisyDense if noisy else Dense
    return Sequential([layer(unit, activation, use_bias=True, kernel_regularizer=_reg,
                             bias_regularizer=_reg) for unit in units])


def make_cnn(filters, kernels, strides, activation='tanh', reg=1e-6):
    _reg = l2(reg)
    cnn = Sequential([Conv2D(f, k, s, activation=activation, kernel_regularizer=reg)
                      for f, k, s in zip(filters, kernels, strides)], name='CNN')
    cnn.add(Flatten())
    return cnn


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


@register("DuelingDQN_uni")
def make_model(name, obs_space, action_space, reg):
    img = dict()
    feat = dict()
    bases = list()
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
    if len(feat) > 0:
        feat_base = concatenate(feat.values())
        bases.append(make_mlp([400, 300], 'tanh', reg)(feat_base))
    if len(img) > 0:
        img_base = concatenate(img.values())
        normalized = img_base/255
        bases.append(make_cnn([32, 32, 32, 32], [3, 3, 3, 3],
                              [2, 2, 2, 2], 'tanh', reg)(normalized))
    base = concatenate(bases)
    head = DuelingModel([512], action_space.n, reg)(base)
    model = tf.keras.Model(inputs={**img, **feat}, outputs=head, name=name)
    return model


@register("DuelingDQN_arm_cube")
def make_model(name, obs_space, action_space, reg=1e-6):
    cube = tf.keras.Input(shape=obs_space['cube'].shape)
    arm = tf.keras.Input(shape=obs_space['arm'].shape)
    features = concatenate([arm, cube])
    base = make_mlp([400, 300], 'tanh', reg)(features)
    head = DuelingModel([512], action_space.n, reg)(base)
    model = tf.keras.Model(inputs={'cube': cube, 'arm': arm}, outputs=head, name=name)
    return model


@register("Critic_uni")
def make_critic(name, obs_space, action_space, reg=1e-6, noisy_head=False):
    img = dict()
    feat = dict()
    bases = list()
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
    feat_base = concatenate(feat.values())

    bases.append(action)
    bases.append(layer(400, 'relu', use_bias=True,
                       kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(feat_base))
    if len(img) > 0:
        img_base = concatenate(img.values())
        normalized = img_base/255
        bases.append(make_cnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2],
                              'tanh', reg)(normalized))
    base = concatenate(bases)
    base = layer(300, 'relu', use_bias=True,  kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)

    head = layer(1, use_bias=True, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    model = tf.keras.Model(inputs={**img, **feat, 'action': action}, outputs=head, name=name)
    return model


@register("Actor_uni")
def make_model(name, obs_space, action_space, reg=1e-6, noisy_head=False):
    img = dict()
    feat = dict()
    bases = list()
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
    if len(feat) > 0:
        feat_base = concatenate(feat.values())
        bases.append(layer(400, 'relu', use_bias=True,
                     kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(feat_base))
    if len(img) > 0:
        img_base = concatenate(img.values())
        normalized = img_base/255
        bases.append(make_cnn([32, 32, 32, 32], [3, 3, 3, 3],
                              [2, 2, 2, 2], 'relu', reg)(normalized))
    base = concatenate(bases)
    base = layer(300, 'relu', use_bias=True,  kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    head = layer(action_space.shape[0], 'tanh', use_bias=True,
                 kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(base)
    head = head * (action_space.high - action_space.low)/2 + (action_space.high + action_space.low)/2
    model = tf.keras.Model(inputs={**img, **feat}, outputs=head, name=name)
    return model