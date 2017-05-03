import os
import tensorflow as tf
import time
import resource
from RHS import RHS
from read_data import Data

rate = 0.00001
epoch_times = 10
loop = 1000


batch_size = 32
segment_per_writer = 50
segment_length = 100
channel = 3


dataset = ['../lfw/train']
log_dir = './log'
model_dir = './model'

rhs = RHS(lstm_size=800, class_num=10)
data = Data(segment_per_writer, segment_length)


def train():

    x = tf.placeholder(tf.float32, shape=(batch_size, segment_length, channel))
    labels = tf.placeholder(tf.int32, shape=(batch_size))
    train_op = rhs.train(rate, x, batch_size, segment_length, labels)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    with sess.as_default():

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)

        if checkpoint:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()

        print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        for step in xrange(loop):
            start_time = time.time()
            print('step: %d' % step)
            x_feed, labels_feed = data.feed_dict(batch_size)
            summary_str, loss = sess.run([summary, train_op], feed_dict={x: x_feed, labels: labels_feed})
            summary_writer.add_summary(summary_str, step)

            if step % 20 == 0:
                checkpoint_file = os.path.join(model_dir, 'model.latest')
                saver.save(sess, checkpoint_file)

            if step % 100 == 0:
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)

            print("step cost: %ds" % (time.time() - start_time))
            print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        summary_writer.close()


if __name__ == '__main__':
    train()
