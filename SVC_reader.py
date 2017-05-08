import os

base_path = './SVC2004/Task1'
useless_line = [0]


def get_writer_list():
    return range(1, 41)


def genuine_data_range():
    return range(1, 21)


def fake_data_range():
    return range(21, 41)


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
                s.append([int(data[0]) - base_x, int(data[1]) - base_y, 1 if int(data[3]) * base_p > 0 else 0])

            base_x = int(data[0])
            base_y = int(data[1])
            base_p = int(data[3])

    except Exception, e:
        print repr(e)
        return None

    return s


def get_genuine_data():
    data = []
    for writer in get_writer_list():
        writer_sample = []
        for index in genuine_data_range():
            writer_sample.append(read_file('{}/U{}S{}.TXT'.format(base_path, writer, index)))
        data.append(writer_sample)
    return data


def get_fake_data():
    data = []
    for writer in get_writer_list():
        writer_sample = []
        for index in fake_data_range():
            writer_sample.append(read_file('{}/U{}S{}.TXT'.format(base_path, writer, index)))
        data.append(writer_sample)
    return data


if __name__ == '__main__':
    # data = read_file('{}/U{}S{}.TXT'.format(base_path, 1, 1))
    data = get_genuine_data()
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))