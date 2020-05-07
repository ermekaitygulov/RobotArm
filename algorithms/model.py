import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, Flatten

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
    def __init__(self, units, action_dim, reg=1e-6):
        super(DuelingModel, self).__init__()
        reg = l2(reg)
        self.h_layers = Sequential([Dense(l, 'relu', kernel_regularizer=reg) for l in units[:-1]])
        self.a_head, self.v_head = Dense(units[-1]/2, 'relu', kernel_regularizer=reg), Dense(units[-1]/2, 'relu', kernel_regularizer=reg)
        self.a_head1, self.v_head1 = Dense(action_dim, kernel_regularizer=reg), Dense(1, kernel_regularizer=reg)

    @tf.function
    def call(self, inputs):
        print('Building model')
        features = self.h_layers(inputs)
        advantage, value = self.a_head(features), self.v_head(features)
        advantage, value = self.a_head1(advantage), self.v_head1(value)
        advantage = advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
        out = value + advantage
        return out


class ClassicCnn(tf.keras.Model):
    def __init__(self, filters, kernels, strides, reg=1e-6):
        super(ClassicCnn, self).__init__()
        reg = l2(reg)
        self.cnn = Sequential(Conv2D(filters[0], kernels[0], strides[0], activation='tanh',
                                     kernel_regularizer=reg), name='CNN')
        for f, k, s in zip(filters[1:], kernels[1:], strides[1:]):
            self.cnn.add(Conv2D(f, k, s, activation='tanh', kernel_regularizer=reg))
        self.cnn.add(Flatten())

    @tf.function
    def call(self, inputs):
        return self.cnn(inputs)


class MLP(tf.keras.Model):
    def __init__(self, units, activation='relu', reg=1e-6):
        super(MLP, self).__init__()
        reg = l2(reg)
        self.model = Sequential([Dense(l, activation, kernel_regularizer=reg) for l in units])

    @tf.function
    def call(self, inputs):
        return self.model(inputs)


@register("DuelingDQN_pov_angle")
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


@register("Critic_pov_angle")
def make_critic(name, obs_space, action_space):
    # TODO add reg
    pov = tf.keras.Input(shape=obs_space['pov'].shape)
    angles = tf.keras.Input(shape=obs_space['angles'].shape)
    action = tf.keras.Input(shape=action_space.shape)
    normalized_pov = pov / 255
    normalized_action = action / 180
    feature_input = tf.keras.layers.concatenate([angles, normalized_action])
    pov_base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])(normalized_pov)
    feature_base = MLP([64, 64], 'tanh')(feature_input)
    base = tf.keras.layers.concatenate([pov_base, feature_base])
    fc = MLP([512, 512], 'relu')(base)
    out = tf.keras.layers.Dense(1)(fc)
    model = tf.keras.Model(inputs={'pov': pov, 'angles': angles, 'action': action},
                           outputs=out, name=name)
    return model


@register("Actor_pov_angle")
def make_actor(name, obs_space, action_space):
    pov = tf.keras.Input(shape=obs_space['pov'].shape)
    angles = tf.keras.Input(shape=obs_space['angles'].shape)
    normalized_pov = pov / 255
    pov_base = ClassicCnn([32, 32, 32, 32], [3, 3, 3, 3], [2, 2, 2, 2])(normalized_pov)
    angles_base = MLP([512, 256], 'tanh')(angles)
    base = tf.keras.layers.concatenate([pov_base, angles_base])
    fc = MLP([512, 512], 'relu')(base)
    out = tf.keras.layers.Dense(action_space.shape[0])(fc)
    out *= 180
    model = tf.keras.Model(inputs={'pov': pov, 'angles': angles}, outputs=out, name=name)
    return model