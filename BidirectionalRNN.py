from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn


class BidirectionalRNN:
    def __init__(self, name, rnn_size, data_type=tf.float32):
        self.data_type = data_type
        self.name = name
        with tf.variable_scope(self.name):
            self.forward_rnn = rnn.MultiRNNCell([rnn.GRUCell(rnn_size[i]) for i in range(len(rnn_size))])
            self.backward_rnn = rnn.MultiRNNCell([rnn.GRUCell(rnn_size[i]) for i in range(len(rnn_size))])

    def run(self, data, reuse=False, time_major=False, pooling=False):
        time_axis = 0 if time_major else 1
        with tf.variable_scope(self.name):
            with tf.variable_scope("ForwardRNN", reuse=reuse) as scope:
                forward_output, state = tf.nn.dynamic_rnn(self.forward_rnn, data, dtype=self.data_type, time_major=time_major, scope=scope)
                if pooling == 'mean':
                    forward_output = tf.reduce_mean(forward_output, time_axis)
                else:
                    forward_output = forward_output[-1, :, :] if time_major else forward_output[:, -1, :]

            with tf.variable_scope("BackwardRNN", reuse=reuse) as scope:
                data = tf.reverse(data, axis=[time_axis])
                backward_output, state = tf.nn.dynamic_rnn(self.backward_rnn, data, dtype=self.data_type, time_major=time_major, scope=scope)
                if pooling == 'mean':
                    backward_output = tf.reduce_mean(backward_output, time_axis)
                else:
                    backward_output = backward_output[-1, :, :] if time_major else backward_output[:, -1, :]

            tf.summary.histogram('forward_rnn_output', forward_output)
            tf.summary.histogram('backward_rnn_output', backward_output)

        return tf.add(forward_output, backward_output, 'feature')
