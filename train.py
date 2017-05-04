import os
import tensorflow as tf
import time
import resource
from RHS import RHS
from read_data import Data

rate = 0.00001
epoch = 10
batch_size = 128
segment_per_sample = 1000
segment_length = 100
channel = 3
test_count = 30


log_dir = './log'
model_dir = './model'

data = Data(segment_per_sample, segment_length)
rhs = RHS(lstm_size=800, class_num=data.class_num())


def train():

    x = tf.placeholder(tf.float32, shape=(batch_size, segment_length, channel))
    labels = tf.placeholder(tf.int32, shape=(batch_size))
    train_op = rhs.train(rate, x, batch_size, segment_length, labels)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()

    with sess.as_default():

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()

        print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        for period in xrange(epoch):
            data.init_data()
            loop = int(data.sample_num() / batch_size)
            for step in xrange(loop):
                global_step = period * loop + step
                start_time = time.time()
                print('epoch: %d step: %d' % (period, step))
                x_feed, labels_feed = data.feed_dict(batch_size)
                summary_str, loss = sess.run([summary, train_op], feed_dict={x: x_feed, labels: labels_feed})
                summary_writer.add_summary(summary_str, global_step)

                if global_step % 20 == 0:
                    checkpoint_file = os.path.join(model_dir, 'model.latest')
                    saver.save(sess, checkpoint_file)

                if global_step % 100 == 0:
                    summary_writer.add_run_metadata(run_metadata, 'step%03d' % global_step)

                print("step cost: %ds" % (time.time() - start_time))
                print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        summary_writer.close()


def test():
    data.init_test_data()
    x = tf.placeholder(tf.float32, shape=(batch_size, segment_length, channel))
    regression = tf.reduce_sum(rhs.run(x, test_count, segment_length))

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
            classification = sess.run(regression, feed_dict={x: writer_sample})
            print (classification)

        print("test cost: %ds" % (time.time() - start_time))


if __name__ == '__main__':
    train()
