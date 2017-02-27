import numpy as np
from sklearn import preprocessing

DATA_SEQUENCE_LEN = 700


def get_seq_lenght(seq_arry, end_symbol):
    """
    return an array of the length of each sequence in seq_arry
    :param seq_arry: array of sequence should be shape of [array_size, max_sequence_length, size_of_symbol]
    :param end_symbol: 1-D array code of the end_symbol
    :return: array of shape [array_size]
    """
    scale_arry = np.argmax(seq_arry, axis=2)
    end_symbol_scale = np.argmax(end_symbol)
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
        "aa_residues": data[:, :, 0:22],
        "ss_label": data[:, :, 22:31],
        "NC_terminals": data[:, :, 31:33],
        # convert to four classes one hot encoding
        "solvent_accessibility": convert_sa_to_one_hot(data[:, :, 33:35]),
        # TODO profile only need 35:56
        "profile": data[:, :, 35:57]
    }
    seq_lens = get_seq_lenght(data_dict["ss_label"], [0] * 8 + [1])

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
    '''
    def next_batch(self, batch_size):
        last_offset = self.offset
        self.offset = (self.offset + batch_size) % (self.num_examples - batch_size)
        # Generate a minibatch.
        batch_data = self.data[self.offset:(self.offset + batch_size), ...]
        batch_labels = self.label[self.offset:(self.offset + batch_size), ...]
        batch_lengths = self.lengths[self.offset:(self.offset + batch_size), ...]
        return batch_data, batch_labels, batch_lengths, last_offset > self.offset
    '''
    def next_batch(self, batch_size):
        last_offset = self.offset
        self.offset = (self.offset + batch_size) % (self.num_examples - batch_size)
        # Generate a minibatch.
        #batch_data = self.data[self.offset:(self.offset + batch_size), ...]
        #batch_labels = self.label[self.offset:(self.offset + batch_size), ...]
        return np.zeros([64, 700, 44], dtype=np.float32), \
            np.zeros([64, 700, 13], dtype=np.float32), \
            np.ones([64, 1])*100,\
            last_offset > self.offset


'''
def get_next_batch(data, input_pl, label_pl):
    batch_data, batch_label = data.next_batch(FLAGS.batch_size)
    fd = {
        input_pl: batch_data,
        label_pl: batch_label
    }
    return fd
'''
