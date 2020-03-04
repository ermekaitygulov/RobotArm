import tensorflow as tf


class DuelingModel(tf.keras.Model):
    def __init__(self, conv_base, units, action_dim):
        super(DuelingModel, self).__init__()
        self.base = conv_base
        self.h_layers = tf.keras.Sequential([tf.keras.layers.Dense(l, activation='relu') for l in units[:-1]])
        self.a_head, self.v_head = tf.keras.layers.Dense(units[-1]/2), tf.keras.layers.Dense(units[-1]/2)
        self.a_head1, self.v_head1 = tf.keras.layers.Dense(action_dim), tf.keras.layers.Dense(1)

    @tf.function
    def call(self, input, training):
        features = self.base(input)
        features = self.h_layers(features)
        advantage, value = self.a_head(features), self.v_head(features)
        advantage, value = self.a_head1(advantage), self.v_head1(value)
        advantage = advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
        out = value + advantage
        return out


def classic_cnn(filters, kernels, strides):
    cnn = tf.keras.Sequential([tf.keras.layers.Conv2D(f, k, s, activation='relu')
                               for f, k, s in zip(filters, kernels, strides)])
    cnn.add(tf.keras.layers.Flatten())
    return cnn
