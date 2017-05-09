from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from RHS import RHS
from SVC_reader import *
import matplotlib.pyplot as plt
import random

channel = 3

log_dir = './log'
model_dir = './model'
sample_count = 2
test_count = 30
segment_length = 100

rhs = RHS(lstm_size=800, class_num=133)
genuine_data = get_genuine_data()
fake_data = get_fake_data()
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(model_dir)


def get_rhs_segment(sample):
    rhs = []
    for i in range(test_count):
        sample_length = len(sample)
        segment_start = random.randint(0, sample_length - segment_length)
        rhs.append(sample[segment_start: segment_start + segment_length])

    return rhs


def test():
    x = tf.placeholder(tf.float32, shape=(test_count, None, channel))
    lstm_code = tf.reduce_sum(rhs.lstm(x, test_count), 0)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # same writer genuine to fake test
        print('same writer genuine to fake test')
        genuine_dis = []
        fake_dis = []
        for writer in range(len(genuine_data)):
            writer_genuine_sample = genuine_data[writer]
            writer_fake_sample = fake_data[writer]
            genuine_lstm_list = []
            fake_lstm_list = []
            base_sample = get_rhs_segment(writer_genuine_sample[0])
            base_lstm = sess.run(lstm_code, feed_dict={x: base_sample})
            for index in range(sample_count):
                index = 2 * index + 3
                genuine_sample = get_rhs_segment(writer_genuine_sample[index])
                fake_sample = get_rhs_segment(writer_fake_sample[index])
                genuine_lstm = sess.run(lstm_code, feed_dict={x: genuine_sample})
                fake_lstm = sess.run(lstm_code, feed_dict={x: fake_sample})
                genuine_lstm_list.append(sess.run(tf.reduce_mean(tf.square(genuine_lstm - base_lstm))))
                fake_lstm_list.append(sess.run(tf.reduce_mean(tf.square(fake_lstm - base_lstm))))

            genuine_dis.append(genuine_lstm_list)
            fake_dis.append(fake_lstm_list)

        genuine_dis = np.array(genuine_dis)
        fake_dis = np.array(fake_dis)
        category = range(len(genuine_data))

        for index in range(sample_count):
            plt.plot(category, genuine_dis[:, index], 'b')
            plt.plot(category, fake_dis[:, index], 'r')
        plt.show()


if __name__ == '__main__':
    test()
