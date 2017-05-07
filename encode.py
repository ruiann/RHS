import tensorflow as tf
import time
import random
from RHS import RHS
from read_data import Data

segment_per_sample = 1000
segment_length = 100
channel = 3
test_count = 30
test_period = 30

log_dir = './log'
model_dir = './model'

data = Data(segment_per_sample, segment_length)
rhs = RHS(lstm_size=800, class_num=data.class_num())


def test():
    data.init_test_data()
    x = tf.placeholder(tf.float32, shape=(test_count, None, channel))
    lstm_code = tf.reduce_sum(rhs.lstm(x, test_count), 0)

    sess = tf.Session()

    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)

        start_time = time.time()
        sample_class = random.randint(0, data.class_num() - 1)

        # sample origin differ
        for period in xrange(test_period):
            sample = data.get_segments_for_each_writer(test_count)
            writer_sample = sample[sample_class]
            lstm = sess.run(lstm_code, feed_dict={x: writer_sample})
            if period > 0:
                differ = lstm - base_lstm
                dis = tf.reduce_mean(tf.square(differ))
                dis = sess.run(dis)
                print(dis)
            base_lstm = lstm

        print("time cost: {0}".format(time.time() - start_time))
        start_time = time.time()
        sample = data.get_segments_for_each_writer(test_count)

        # different origin differ
        for period in xrange(sample_class):
            writer_sample = sample[period]
            lstm = sess.run(lstm_code, feed_dict={x: writer_sample})
            if period > 0:
                differ = lstm - base_lstm
                dis = tf.reduce_mean(tf.square(differ))
                dis = sess.run(dis)
                print(dis)
            base_lstm = lstm

        print("time cost: {0}".format(time.time() - start_time))


if __name__ == '__main__':
    test()
