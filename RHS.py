# definition RHS model

import tensorflow as tf
from LogisticRegression import LogisticRegression
from BidirectionalLSTM import BidirectionalLSTM


class RHS:

    def __init__(self, lstm_size=800, class_num=10):
        self.bidirectional_LSTM = BidirectionalLSTM('BidirectionalLSTM', lstm_size)
        self.logistic_regression = LogisticRegression('LogisticRegression', lstm_size, class_num)

    # do classification
    def run(self, data, batch_size, length):
        lstm_code = self.bidirectional_LSTM.run(data, batch_size, length)
        return self.logistic_regression.run(lstm_code)

    # compute loss
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='RHS'))

    # return training operation, data should be a PlaceHolder
    def train(self, rate, data, batch_size, length, labels):
        logits = self.run(data, batch_size, length)
        loss = self.loss(logits, labels)
        tf.summary.scalar('classifier loss', loss)
        return tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
