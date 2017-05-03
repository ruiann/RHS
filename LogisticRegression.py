# definition the logistic regression for classification

import tensorflow as tf


class LogisticRegression:

    def __init__(self, name, input_dimension, output_dimension):
        self.name = name
        self.W, self.b = self.full_connection_layer(input_dimension, output_dimension)

    def full_connection_layer(self, input_dim, out_dim):
        with tf.variable_scope(self.name):
            W = tf.get_variable('weight', dtype=tf.float32, shape=[input_dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('bias', dtype=tf.float32, shape=[out_dim], initializer=tf.constant_initializer(0.0))
        return W, b

    def run(self, x):
        out = tf.nn.bias_add(tf.matmul(x, self.W), self.b)
        out = tf.nn.relu(out)
        return out
