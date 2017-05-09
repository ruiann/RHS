from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from RHS import RHS
from ATVS_reader import *
import matplotlib.pyplot as plt
import random

channel = 3

log_dir = './log'
model_dir = './model'

rhs = RHS(lstm_size=800, class_num=133)
genuine_data = get_genuine_data()
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(model_dir)


def test():
    x = tf.placeholder(tf.float32, shape=(1, None, channel))
    lstm_code = tf.reduce_sum(rhs.lstm(x, 1), 0)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # different writer genuine test
        print('different writer genuine test')
        start_time = time.time()
        writer = random.randint(0, len(genuine_data))
        writer_genuine_sample = genuine_data[writer]
        genuine_lstm_dis = []
        base_sample = writer_genuine_sample[0]
        base_lstm = sess.run(lstm_code, feed_dict={x: [base_sample]})
        for index in xrange(5):
            index = random.randint(1, len(writer_genuine_sample) - 1)
            sample = writer_genuine_sample[index]
            lstm = sess.run(lstm_code, feed_dict={x: [sample]})
            genuine_lstm_dis.append(sess.run(tf.reduce_mean(tf.square(lstm - base_lstm))))
        print("time cost: {0}".format(time.time() - start_time))

        for sample in genuine_lstm_dis:
            plt.plot(sample, 'b', linewidth=1)
        plt.show()


if __name__ == '__main__':
    test()
