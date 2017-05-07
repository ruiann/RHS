# definition RHS model

import tensorflow as tf
from LogisticRegression import LogisticRegression
from BidirectionalLSTM import BidirectionalLSTM


class RHS:

    def __init__(self, lstm_size=800, class_num=10):
        self.bidirectional_LSTM = BidirectionalLSTM('BidirectionalLSTM', lstm_size)
        self.logistic_regression = LogisticRegression('LogisticRegression', lstm_size, class_num)

    # do classification
    def run(self, data, batch_size):
        lstm_code = self.lstm(data, batch_size)
        return self.regression(lstm_code)

    def lstm(self, data, batch_size):
        return self.bidirectional_LSTM.run(data, batch_size)

    def regression(self, lstm_code):
        return self.logistic_regression.run(lstm_code)

    # compute loss
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='RHS'))

    # return training operation, data should be a PlaceHolder
    def train(self, rate, data, batch_size, labels):
        logits = self.run(data, batch_size)
        classification = tf.to_int32(tf.arg_max(tf.nn.softmax(logits), dimension=1))
        differ = labels - classification
        tf.summary.histogram('classification difference', differ)
        loss = self.loss(logits, labels)
        tf.summary.scalar('classifier loss', loss)
        return tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
