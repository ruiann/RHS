# definition RHS model

import tensorflow as tf
from LogisticRegression import LogisticRegression
from BidirectionalRNN import BidirectionalRNN


class RHS:

    def __init__(self, rnn_size=[800], layer=[10]):
        self.bidirectional_rnn = BidirectionalRNN('BidirectionalRNN', rnn_size)
        self.logistic_regression = LogisticRegression('LogisticRegression', rnn_size[-1], layer)

    # do classification
    def run(self, data, batch_size):
        rnn_code = self.rnn(data, batch_size)
        return self.regression(rnn_code)

    def rnn(self, data, batch_size):
        return self.bidirectional_rnn.run(data, batch_size)

    def regression(self, rnn_code):
        return self.logistic_regression.run(rnn_code)

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
