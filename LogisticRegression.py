from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LogisticRegression:

    def __init__(self, name, input_dimension, layer=[1]):
        self.name = name
        self.input_dimension = input_dimension
        self.layer = layer

    def full_connection_layer(self, index, input_dim, out_dim):
        with tf.variable_scope('{}_layer_{}'.format(self.name, index + 1)):
            W = tf.get_variable('weight', dtype=tf.float32, shape=[input_dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('bias', dtype=tf.float32, shape=[out_dim], initializer=tf.constant_initializer(0.0))
        return W, b

    def run(self, x):
        out = x
        for i in range(len(self.layer)):
            if i == 0:
                W, b = self.full_connection_layer(i, self.input_dimension, self.layer[i])
            else:
                W, b = self.full_connection_layer(i, self.layer[i - 1], self.layer[i])
            out = tf.nn.bias_add(tf.matmul(out, W), b)
            if i != len(self.layer) - 1:
                out = tf.nn.relu(out)
            tf.summary.histogram('regression_result_layer {}'.format(i + 1), out)
        return out
