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
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(model_dir)


def test():
    x = tf.placeholder(tf.float32, shape=(1, None, channel))
    lstm_code = tf.reduce_sum(rhs.lstm(x, 1), 0)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint.model_checkpoint_path)

        for writer in xrange(len(genuine_data)):
            writer_sample = genuine_data[writer]
            for index in xrange(len(writer_sample)):
                sample = writer_sample[index]
                start_time = time.time()
                lstm = sess.run(lstm_code, feed_dict={x: [sample]})
                if index > 0:
                    differ = lstm - base_lstm
                    dis = tf.reduce_mean(tf.square(differ))
                    dis = sess.run(dis)
                    print(dis)
                base_lstm = lstm
                print("time cost: {0}".format(time.time() - start_time))


if __name__ == '__main__':
    test()
