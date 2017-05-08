import tensorflow as tf
import time
from RHS import RHS
from SVC_reader import *
import pdb

channel = 3

log_dir = './log'
model_dir = './model'

rhs = RHS(lstm_size=800, class_num=133)
genuine_data = get_genuine_data()
fake_data = get_fake_data()
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(model_dir)


def test():
    x = tf.placeholder(tf.float32, shape=(1, None, channel))
    lstm_code = tf.reduce_sum(rhs.lstm(x, 1), 0)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # same writer genuine test
        print('same writer genuine test')
        start_time = time.time()
        for writer in xrange(len(genuine_data)):
            writer_sample = genuine_data[writer]
            dis = []
            for index in xrange(len(writer_sample)):
                sample = writer_sample[index]
                lstm = sess.run(lstm_code, feed_dict={x: [sample]})
                if index > 0:
                    differ = lstm - base_lstm
                    distance = tf.reduce_sum(tf.square(differ))
                    dis.append(sess.run(distance))
                base_lstm = lstm

            min, max, mean = sess.run([tf.reduce_min(dis), tf.reduce_max(dis), tf.reduce_mean(dis)])
            print('min: {}, max: {}, mean: {}'.format(min, max, mean))
        print("time cost: {0}".format(time.time() - start_time))

        # same writer genuine to fake test
        print('same writer genuine to fake test')
        start_time = time.time()
        for writer in xrange(len(genuine_data)):
            writer_genuine_sample = genuine_data[writer]
            writer_fake_sample = fake_data[writer]
            dis = []
            for index in xrange(len(writer_genuine_sample)):
                genuine_sample = writer_genuine_sample[index]
                fake_sample = writer_fake_sample[index]
                genuine_lstm = sess.run(lstm_code, feed_dict={x: [genuine_sample]})
                fake_lstm = sess.run(lstm_code, feed_dict={x: [fake_sample]})
                differ = fake_lstm - genuine_lstm
                distance = tf.reduce_sum(tf.square(differ))
                dis.append(sess.run(distance))

            min, max, mean = sess.run([tf.reduce_min(dis), tf.reduce_max(dis), tf.reduce_mean(dis)])
            print('min: {}, max: {}, mean: {}'.format(min, max, mean))
        print("time cost: {0}".format(time.time() - start_time))


if __name__ == '__main__':
    test()
