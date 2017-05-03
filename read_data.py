import random

# data definition of sigComp2011
def read_file(path):
    try:
        file = open(path, 'r')
        lines = file.readlines()
        s = []
        base_x = None
        base_y = None
        base_p = None
        for line in lines:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            data = line.split(' ')

            if base_x:
                s.append([int(data[0]) - base_x, int(data[1]) - base_y, 1 if int(data[2]) * base_p > 0 else 0])

            base_x = int(data[0])
            base_y = int(data[1])
            base_p = int(data[2])

    except:
        return None

    return s

# genuine sample for sigComp2011

writer_num = 10
sample = 24
dir = './TrainingSet/Online Genuine/'

def get_samples():
    samples = []
    for w in xrange(writer_num):
        signatures = []
        no = str(w + 1)
        prefix = '00' + no if len(no) == 1 else '0' + no
        for s in xrange(sample):
            path = '{}{}_{}.HWR'.format(dir, prefix, s + 1)
            signature = read_file(path)
            if signature:
                signatures.append(signature)
        samples.append(signatures)

    return samples


# get rhs data
def get_rhs_segments(segment_per_writer=50, segment_length=100):
    samples = get_samples()
    rhs = []
    for w in xrange(writer_num):
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
    def __init__(self, segment_per_writer, segment_length):
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