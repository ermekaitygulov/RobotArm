import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.regularizers import l2


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

    def call(self, inputs, training=None, mask=None):
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
        self.input_dim = None
        self.kernel_sigma = None
        self.bias_sigma = None
        super(NoisyDense, self).__init__(units, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                            initializer=self.kernel_initializer,
                                            name='sigma_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=None)

        if self.use_bias:
            self.bias_sigma = self.add_weight(shape=(1, self.units,),
                                              initializer=self.bias_initializer,
                                              name='bias_sigma',
                                              regularizer=self.bias_regularizer,
                                              constraint=None)
        else:
            self.bias_sigma = None
        super(NoisyDense, self).build(input_shape)

    def call(self, inputs, training=False):
        if inputs.shape[0]:
            kernel_input = self.f(tf.random.normal(shape=(inputs.shape[0], self.input_dim, 1)))
            kernel_output = self.f(tf.random.normal(shape=(inputs.shape[0], 1, self.units)))
        else:
            kernel_input = self.f(tf.random.normal(shape=(self.input_dim, 1)))
            kernel_output = self.f(tf.random.normal(shape=(1, self.units)))
        kernel_epsilon = tf.matmul(kernel_input, kernel_output)

        w = self.kernel + self.kernel_sigma * kernel_epsilon

        output = tf.matmul(tf.expand_dims(inputs, axis=1), w)

        if self.use_bias:
            b = self.bias + self.bias_sigma * kernel_output
            output = output + b
        if self.activation is not None:
            output = self.activation(output)
        output = tf.squeeze(output, axis=1)
        return output


class HistogramLayer(tf.keras.layers.Layer):
    def __init__(self, name):
        super(HistogramLayer, self).__init__()
        self.step = 0
        self.hist_name = name

    def call(self,  inputs, training=None, mask=None):
        tf.summary.histogram('output_hist/{}'.format(self.hist_name), inputs, self.step)
        self.step += 1
        return inputs


class TwinCritic(tf.keras.Model):
    def __init__(self, build_function, name, obs_space, action_space):
        super(TwinCritic, self).__init__()
        self.twin1 = build_function(name + '_1', obs_space, action_space)
        self.twin2 = build_function(name + '_2', obs_space, action_space)

    def call(self,  inputs, training=None, mask=None):
        return self.twin1(inputs, training=training, mask=mask),\
               self.twin2(inputs, training=training, mask=mask)


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel, reg, use_bn=False):
        super(ResnetIdentityBlock, self).__init__(name='')

        self.conv2a = tf.keras.layers.Conv2D(filters, kernel, padding='same',
                                             kernel_regularizer=reg, bias_regularizer=reg)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters, kernel, padding='same',
                                             kernel_regularizer=reg, bias_regularizer=reg)
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.use_bn = use_bn

    def call(self,  inputs, training=None, mask=None):
        x = self.conv2a(inputs)
        if self.use_bn:
            x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        if self.use_bn:
            x = self.bn2b(x, training=training)

        x += inputs
        return tf.nn.relu(x)


def make_twin(build_critic):
    def thunk(name, obs_space, action_space):
        return TwinCritic(build_critic, name, obs_space, action_space)
    return thunk


def make_mlp(units, activation='relu', reg=1e-6, noisy=False):
    _reg = l2(reg)
    layer = NoisyDense if noisy else Dense
    return Sequential([layer(unit, activation, kernel_regularizer=_reg,
                             bias_regularizer=_reg) for unit in units])


def make_cnn(filters, kernels, strides, activation='tanh', reg=1e-6, flat=False):
    _reg = l2(reg)
    cnn = Sequential([Conv2D(f, k, s, activation=activation, kernel_regularizer=_reg)
                      for f, k, s in zip(filters, kernels, strides)])
    if flat:
        cnn.add(Flatten())
    return cnn


def make_impala_cnn(filters=(16, 32, 32), reg=1e-6, flat=False, use_bn=False):
    _reg = l2(reg)
    cnn = Sequential()
    for f in filters:
        cnn.add(Conv2D(f, 3, 2, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
        cnn.add(ResnetIdentityBlock(f, 3, l2(reg), use_bn))
    if flat:
        cnn.add(Flatten())
    return cnn
