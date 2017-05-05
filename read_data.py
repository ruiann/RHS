import random
import os

base_path = './data'
train_dir = ['chinese1', 'chinese2', 'english1', 'english2']
test_dir = ['same chinese', 'same english']
useless_line = (0, 0, 0, 0, -1)


def get_writer_list():
    filenames = os.listdir(base_path + '/' + train_dir[0])
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
def get_samples(dir_list):
    writers = get_writer_list()
    samples = []
    for w in xrange(len(writers)):
        signatures = []
        no = str(w + 1)
        for s in xrange(len(dir_list)):
            path = '{}/{}/{}'.format(base_path, dir_list[s], writers[w])
            signature = read_file(path)
            if signature:
                signatures.append(signature)
        samples.append(signatures)

    return samples, writers


# get rhs data
def get_rhs_segments(dir_list, segment_per_sample=1000, segment_length=100):
    samples, writers = get_samples(dir_list)
    rhs = []
    for w in xrange(len(writers)):
        writer_samples = samples[w]
        count = len(writer_samples)
        for sample_index in xrange(count):
            for i in xrange(segment_per_sample):
                sample = writer_samples[sample_index]
                sample_length = len(sample)
                segment_start = random.randint(0, sample_length - segment_length)
                rhs.append({'segment': sample[segment_start: segment_start + segment_length], 'label': w})

    return rhs


class Data:
    def __init__(self, segment_per_sample=1000, segment_length=100):
        self.segment_per_sample = segment_per_sample
        self.segment_length = segment_length
        self.writers = get_writer_list()

    def init_data(self):
        self.rhs_sample = get_rhs_segments(train_dir, self.segment_per_sample, self.segment_length)

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

    def class_num(self):
        return len(self.writers)

    def sample_num(self):
        return len(self.rhs_sample)

    def init_test_data(self):
        test_rhs_sample = get_rhs_segments(test_dir, self.segment_per_sample, self.segment_length)
        result = [[] for i in xrange(self.class_num())]
        for sample in test_rhs_sample:
            label = sample['label']
            result[label].append(sample['segment'])

        self.test_rhs_sample = result

    def get_segments_for_each_writer(self, count=2):
        result = []
        for i in xrange(self.class_num()):
            writer_result = []
            for j in xrange(count):
                writer_sample = self.test_rhs_sample[i]
                segments_count = len(writer_sample)
                segment_index = random.randint(0, segments_count - 1)
                writer_result.append(writer_sample[segment_index])
            result.append(writer_result)

        return result

if __name__ == '__main__':
    data = Data()
    samples, labels = data.feed_dict(10)
    print samples
    print labels
