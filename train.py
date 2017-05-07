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

log_dir = './log'
model_dir = './model'

data = Data(segment_per_sample, segment_length)
rhs = RHS(lstm_size=800, class_num=data.class_num())


def train():

    x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    labels = tf.placeholder(tf.int32, shape=(batch_size))
    train_op = rhs.train(rate, x, batch_size, labels)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(model_dir)

    with sess.as_default():

        sess.run(tf.global_variables_initializer())

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
                print('epoch: {0} step: {1}'.format(period, step))
                x_feed, labels_feed = data.feed_dict(batch_size)
                summary_str, loss = sess.run([summary, train_op], feed_dict={x: x_feed, labels: labels_feed})
                summary_writer.add_summary(summary_str, global_step)

                if global_step % 20 == 0 and global_step != 0:
                    checkpoint_file = os.path.join(model_dir, 'model.latest')
                    saver.save(sess, checkpoint_file)

                if global_step % 100 == 0:
                    summary_writer.add_run_metadata(run_metadata, 'step%03d' % global_step)

                print("step cost: {0}".format(time.time() - start_time))
                print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        summary_writer.close()


if __name__ == '__main__':
    train()
