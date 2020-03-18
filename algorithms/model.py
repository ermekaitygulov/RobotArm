import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, Flatten


class DuelingModel(tf.keras.Model):
    def __init__(self, units, action_dim):
        super(DuelingModel, self).__init__()
        self.h_layers = Sequential([Dense(l, activation='relu', kernel_regularizer=l2(1e-5)) for l in units[:-1]])
        self.a_head, self.v_head = Dense(units[-1]/2, kernel_regularizer=l2(1e-5)), Dense(units[-1]/2, kernel_regularizer=l2(1e-5))
        self.a_head1, self.v_head1 = Dense(action_dim, kernel_regularizer=l2(1e-5)), Dense(1, kernel_regularizer=l2(1e-5))

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
    def __init__(self, filters, kernels, strides):
        super(ClassicCnn, self).__init__()
        self.cnn = Sequential(Conv2D(filters[0], kernels[0], strides[0],
                                            activation='relu'), name='CNN')
        for f, k, s in zip(filters[1:], kernels[1:], strides[1:]):
            self.cnn.add(Conv2D(f, k, s, activation='relu'))
        self.cnn.add(Flatten())

    @tf.function
    def call(self, inputs):
        return self.cnn(inputs)
