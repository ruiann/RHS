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
rhs = RHS(lstm_size=800, class_num=data.class_num())


def test():
    data.init_test_data()
    x = tf.placeholder(tf.float32, shape=(test_count, segment_length, channel))
    regression = rhs.run(x, test_count, segment_length)
    classification = tf.reduce_mean(tf.nn.softmax(regression), 0)

    sess = tf.Session()

    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)

        start_time = time.time()
        sample = data.get_segments_for_each_writer(test_count)

        for w in xrange(len(sample)):
            writer_sample = sample[w]
            probability = sess.run(classification, feed_dict={x: writer_sample})
            print probability

        print("test cost: %ds" % (time.time() - start_time))


if __name__ == '__main__':
    test()