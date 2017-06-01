from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from RHS import RHS
from read_data import Data

segment_per_sample = 1000
segment_length = 100
channel = 3
test_count = 30

model_dir = './model'

data = Data(segment_per_sample, segment_length)
rhs = RHS(layer=[data.class_num()])


def test():
    data.init_test_data()
    x = tf.placeholder(tf.float32, shape=(test_count, None, channel))
    lstm_code = rhs.lstm(x, test_count)
    regression = rhs.regression(lstm_code)
    classification = tf.reduce_mean(tf.nn.softmax(regression), 0)
    index = tf.argmax(classification, dimension=0)
    sample_code = tf.reduce_sum(lstm_code, 0)

    sess = tf.Session()

    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)

        sample = data.get_segments_for_each_writer(test_count)

        # classification test
        label_list = []
        start_time = time.time()
        for w in xrange(len(sample)):
            writer_sample = sample[w]
            probability, label = sess.run([classification, index], feed_dict={x: writer_sample})
            print('label: {0} probability: {1}'.format(label, probability[label]))
            label_list.append(label)

        print("test cost: {0}".format(time.time() - start_time))
        print(label_list)

        # different writer encoding distance
        start_time = time.time()
        dis = []
        for w in xrange(len(sample)):
            writer_sample = sample[w]
            lstm = sess.run(sample_code, feed_dict={x: writer_sample})
            if w == 0:
                base_lstm = lstm
            else:
                dis.append(sess.run(tf.reduce_mean(tf.square(lstm - base_lstm))))

        print("test cost: {0}".format(time.time() - start_time))
        print(dis)

        # same writer encoding distance
        start_time = time.time()
        dis = []
        for w in xrange(len(sample)):
            for index in range(2):
                sample = data.get_segments_for_each_writer(test_count)
                writer_sample = sample[w]
                lstm = sess.run(sample_code, feed_dict={x: writer_sample})
                if index == 0:
                    base_lstm = lstm
                else:
                    dis.append(sess.run(tf.reduce_mean(tf.square(lstm - base_lstm))))

        print("test cost: {0}".format(time.time() - start_time))
        print(dis)


if __name__ == '__main__':
    test()
