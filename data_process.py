import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tqdm import tqdm
import os
import pickle as pk
DATA_SEQUENCE_LEN = 700

def split_and_convert(file_name):
    data = read_data_from_example(file_name)
    train_data, valid_data, _ = get_train_valid_test(data, [0.85, 0.15, 0])
    record_prefix = os.path.splitext(file_name)[0]

    # save training data
    train_file = record_prefix + "_train.tfrecords"
    convert_ICML2014_to_record(train_data, train_file)

    # save valid data
    valid_file = record_prefix + "_valid.pkl"
    fh = open(valid_file, "wb")
    pk.dump(valid_data, fh)
    fh.close()


def convert_ICML2014_to_record(input_data, file_name):
    """
    convert ICML2014 dataset to tensorflow binary format
    :param file_name_queue:
    :return:
    """
    all_data = input_data[0]
    all_labels = input_data[1]
    all_length = input_data[2]
    r_index = list(range(len(all_data)))
    np.random.shuffle(r_index)

    writer = tf.python_io.TFRecordWriter(file_name)

    for idx in tqdm(r_index):
        label = np.reshape(all_labels[idx, ...], [-1])
        data = np.reshape(all_data[idx, ...], [-1])
        lengths = all_length[idx]
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
                # Features contains a map of string to Feature proto objects
                feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=label.astype(np.int64))),
                    'data': tf.train.Feature(
                        float_list=tf.train.FloatList(value=data.astype(np.float64))),
                    'length': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[lengths]))
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)


def read_record_file_for_test(file_name):
    i = 0
    for serialized_example in tf.python_io.tf_record_iterator(file_name):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # traverse the Example format to get data
        length = example.features.feature['length'].int64_list.value[0]
        data = example.features.feature['data'].float_list.value[:]
        label = example.features.feature['label'].int64_list.value[:]
        # do something
        #print(length)
        #print(label)
        #print(data)
        i += 1
    print(i)






def get_seq_lenght(seq_arry, end_symbol):
    """
    return an array of the length of each sequence in seq_arry
    :param seq_arry: array of sequence should be shape of [array_size, max_sequence_length, size_of_symbol]
    :param end_symbol: 1-D array code of the end_symbol
    :return: array of shape [array_size]
    """
    scale_arry = np.argmax(seq_arry, axis=2) + np.sum(seq_arry, axis=2)
    end_symbol_scale = np.argmax(end_symbol) + np.sum(end_symbol)
    cond = (scale_arry != end_symbol_scale).astype(np.int)
    lens = cond.sum(axis=1)
    return lens


def convert_sa_to_one_hot(sa_array):
    """
    transfer the separate one hot encoding for relative and absolute accessibility like [1, 1], [1,0]
    to four bit one hot label like [0,0,0,1], [1, 0, 0, 1]
    :param sa_array:
    :return:
    """
    # flatten
    sa_array = np.reshape(sa_array, [-1, 2, 1])
    # combine two features
    index = np.multiply(sa_array[:, 0], 2) + sa_array[:, 1]
    #np.reshape(index, (index.shape[0], 1))

    # transfer to one hot
    enc = preprocessing.OneHotEncoder(sparse=False)
    enc.fit([[1], [2], [3], [4]])
    one_hot_code = enc.transform(index)

    # reshape back to shape [..., sequence_length, 4]
    one_hot_code = np.reshape(one_hot_code, [-1, DATA_SEQUENCE_LEN, 4])
    return one_hot_code

def read_data_from_example(file_name):
    """

    :param file_name: read matrix from file_name (.npy file)
    :return:  data, label, lengths for each example
    """
    data = np.load(file_name)
    data = np.reshape(data, [-1, 700, 57])
    data_dict = {
        "aa_residues": data[:, :, 0:21],
        "ss_label": data[:, :, 22:30],
        "NC_terminals": data[:, :, 31:33],
        # convert to four classes one hot encoding
        "solvent_accessibility": convert_sa_to_one_hot(data[:, :, 33:35]),
        "profile": data[:, :, 35:56]
    }
    seq_lens = get_seq_lenght(data_dict["ss_label"], [0] * 8)

    return np.concatenate((data_dict["aa_residues"], data_dict["profile"]), axis=2), \
           np.concatenate((data_dict["ss_label"], data_dict["solvent_accessibility"]), axis=2), \
           seq_lens


def get_train_valid_test(data, ratios):
    """
    split data into training validation and testing set
    according to the ratios
    :param data:
    :param ratios:
    :return:
    """
    data_size = len(data[0])
    train_index = int(data_size * ratios[0])
    valid_index = int(data_size * ratios[1]) + train_index
    test_index = int(data_size * ratios[2]) + valid_index

    indices = np.arange(data_size)
    np.random.shuffle(indices)
    train_data = [item[indices[0:train_index], ...] for item in data]
    valid_data = [item[indices[train_index:valid_index], ...] for item in data]
    test_data = [item[indices[valid_index:test_index], ...] for item in data]

    return train_data, valid_data, test_data


class DataSet(object):
    def __init__(self, data, label, lengths):
        if data.shape[0] != label.shape[0]:
            raise ValueError("data and label must have same length")
        if (data.shape[1] != DATA_SEQUENCE_LEN) or (label.shape[1] != DATA_SEQUENCE_LEN):
            raise ValueError("data and label sequence length not equals to:", DATA_SEQUENCE_LEN)
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)
        self.lengths = lengths.astype(np.int32)
        self.num_examples = label.shape[0]
        self.offset = 0

    def next_batch(self, batch_size):
        last_offset = self.offset
        next_offset = (self.offset + batch_size)
        if next_offset > self.num_examples:
            batch_size = self.num_examples - self.offset
            next_offset = 0
        # Generate a minibatch.
        batch_data = self.data[self.offset:(self.offset + batch_size), ...]
        batch_labels = self.label[self.offset:(self.offset + batch_size), ...]
        batch_lengths = self.lengths[self.offset:(self.offset + batch_size), ...]
        self.offset = next_offset
        return batch_data, batch_labels, batch_lengths, last_offset > self.offset
