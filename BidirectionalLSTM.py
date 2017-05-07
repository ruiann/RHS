# define the bidirectional LSTM network for RHS feature extraction

import tensorflow as tf


class BidirectionalLSTM:

    def __init__(self, name, lstm_size, data_type=tf.float32):
        self.data_type = data_type
        self.name = name
        with tf.variable_scope(self.name):
            self.forward_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.backward_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    def clear(self, batch_size):
        return self.forward_lstm.zero_state(batch_size, self.data_type), self.backward_lstm.zero_state(batch_size, self.data_type)

    def run(self, data, batch_size):
        forward_initial_state, backward_initial_state = self.clear(batch_size)

        with tf.variable_scope("ForwardLSTM"):
            state = forward_initial_state
            forward_output, state = tf.nn.dynamic_rnn(self.forward_lstm, data, initial_state=state)
            forward_output = forward_output[:, -1, :]

        with tf.variable_scope("BackwardLSTM"):
            state = backward_initial_state
            backward_output, state = tf.nn.dynamic_rnn(self.forward_lstm, data, initial_state=state)
            backward_output = backward_output[:, -1, :]

        tf.summary.histogram('forward_lstm_output', forward_output)
        tf.summary.histogram('backward_lstm_output', backward_output)

        return tf.add(forward_output, backward_output, 'feature')
