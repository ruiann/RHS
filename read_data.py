import random
import os
import pdb


train_dir = ['./data/chinese1', './data/chinese2', './data/english1', './data/english2']
test_dir = ['./data/same chinese', './data/same english']
useless_line = (0, 0, 0, 0, -1)


def get_writter_list():
    filenames = os.listdir(train_dir[0])
    return filenames


# data definition of BIT Handwriting
def read_file(path):
    try:
        file = open(path, 'r')
        lines = file.readlines()
        for line_index in useless_line:
            del lines[line_index]
        s = []
        base_x = None
        base_y = None
        base_p = None
        for line in lines:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            data = line.split()
            if base_x:
                s.append([int(data[4]) - base_x, int(data[5]) - base_y, 1 if int(data[1]) * base_p > 0 else 0])

            base_x = int(data[4])
            base_y = int(data[5])
            base_p = int(data[1])

    except Exception, e:
        print repr(e)
        return None

    return s


# sample for BIT Handwriting
def get_samples():
    writters = get_writter_list()
    samples = []
    for w in xrange(len(writters)):
        signatures = []
        no = str(w + 1)
        for s in xrange(len(train_dir)):
            path = '{}/{}'.format(train_dir[s], writters[w])
            signature = read_file(path)
            if signature:
                signatures.append(signature)
        samples.append(signatures)

    return samples, writters


# get rhs data
def get_rhs_segments(segment_per_writer=1000, segment_length=100):
    samples, writters = get_samples()
    rhs = []
    for w in xrange(len(writters)):
        writer_samples = samples[w]
        count = len(writer_samples)
        for i in xrange(segment_per_writer):
            sample_index = random.randint(0, count - 1)
            sample = writer_samples[sample_index]
            sample_length = len(sample)
            segment_start = random.randint(0, sample_length - segment_length)
            rhs.append({'segment': sample[segment_start: segment_start + segment_length], 'label': w})

    return rhs


class Data:
    def __init__(self, segment_per_writer=1000, segment_length=100):
        self.rhs_sample = get_rhs_segments(segment_per_writer, segment_length)

    def feed_dict(self, batch_size):
        segments = []
        labels = []
        for i in xrange(batch_size):
            segments_count = len(self.rhs_sample)
            segment_index = random.randint(0, segments_count - 1)
            segments.append(self.rhs_sample[segment_index]['segment'])
            labels.append(self.rhs_sample[segment_index]['label'])
            del self.rhs_sample[segment_index]

        return segments, labels


if __name__ == '__main__':
    data = Data()
    samples, labels = data.feed_dict(10)
    print samples
    print labels
