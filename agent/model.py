import numpy as np

import tensorflow as tf


class DuelingModel:
    def __init__(self, conv_base, layers_size, action_dim, obs_dim, sess, name):
        self.name = name
        self.conv_base = conv_base
        self.layers_size = layers_size
        self.action_dim = action_dim
        self.sess = sess
        self._step_pov = tf.compat.v1.placeholder("float", [None, ] + obs_dim)
        self._value = self.net(self._step_pov)
        self.params = tf.compat.v1.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

    def net(self, pov_ph):
        with tf.variable_scope(self.name, reuse=True):
            feature_layer = self.conv_base(pov_ph)
            for size in self.layers_size:
                feature_layer = tf.layers.dense(feature_layer, size)
            a_layer, v_layer = tf.split(feature_layer, num_or_size_splits=2, axis=-1, name='stream_split')
            a_layer = tf.layers.dense(a_layer, self.action_dim, name='A')
            a_layer = tf.subtract(a_layer, tf.reduce_mean(a_layer, axis=-1, keepdims=True), name='A')
            v_layer = tf.layers.dense(v_layer, 1, name='V')
            out = tf.add(a_layer, v_layer, name='out')
            return out

    def compute_smth(self, state, smth):
        feed_dict = {self._step_pov: [np.array(state) / 256, ]}
        value = self.sess.run(smth, feed_dict=feed_dict)
        return value

    def value(self, state):
        return self.compute_smth(state, self._value)
