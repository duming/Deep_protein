import numpy as np
import tensorflow as tf
from sklearn import preprocessing
########################
# file input output
########################


class NetConfig(object):
    def __init__(self):
        self.regu_coef = 0.0001
        self.seq_len = 700
        '''
        self.out_first_size = 9
        self.out_second_size = 4
        '''
        self.label_first_size = 9
        self.label_second_size = 4

        self.label_size = self.label_first_size + self.label_second_size
        # preprocess
        self.in_first_size = 22
        self.in_second_size = 22
        self.in_size = self.in_first_size + self.in_second_size
        self.embed_size = 50

        #multiscal convolution
        self.kernel1 = [3, self.embed_size + self.in_second_size, 64]
        self.kernel2 = [7, self.embed_size + self.in_second_size, 64]
        self.kernel3 = [11, self.embed_size + self.in_second_size, 64]

        #fully connected
        self.fc1 = [self.kernel1[-1] + self.kernel2[-1] + self.kernel3[-1], 64]
        self.fc2 = [self.fc1[-1], self.label_size]


net_config = NetConfig()


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_path', '/Users/ming/projects/Deep_protein/exp_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('epoch_num', 10000,
                            """Number of epoch to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer("batch_size", 64,
                            "number of batches")


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
    one_hot_code = np.reshape(one_hot_code, [-1, net_config.seq_len, 4])
    return one_hot_code


def read_data_from_example(file_name):
    """

    :param file_name: read matrix from file_name (.npy file)
    :return:  data, label
    """
    data = np.load(file_name)
    data = np.reshape(data, [-1, 700, 57])
    data_dict = {
        "aa_residues": data[:, :, 0:22],
        "ss_label": data[:, :, 22:31],
        "NC_terminals": data[:, :, 31:33],
        # convert to four classes one hot encoding
        "solvent_accessibility": convert_sa_to_one_hot(data[:, :, 33:35]),
        "profile": data[:, :, 35:57]
    }

    return np.concatenate((data_dict["aa_residues"], data_dict["profile"]), axis=2),\
        np.concatenate((data_dict["ss_label"], data_dict["solvent_accessibility"]), axis=2)


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
    def __init__(self, data, label):
        if data.shape[0] != label.shape[0]:
            raise ValueError("data and label must have same length")
        if (data.shape[1] != net_config.seq_len) or (label.shape[1] != net_config.seq_len):
            raise ValueError("data and label sequence length not equals to:", net_config.seq_len)
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)
        self.num_examples = label.shape[0]
        self.offset = 0

    def next_batch(self, batch_size):
        last_offset = self.offset
        self.offset = (self.offset + batch_size) % (self.num_examples - batch_size)
        # Generate a minibatch.
        batch_data = self.data[self.offset:(self.offset + batch_size), ...]
        batch_labels = self.label[self.offset:(self.offset + batch_size), ...]
        return batch_data, batch_labels, last_offset < self.offset


def get_next_batch(data, input_pl, label_pl):
    batch_data, batch_label = data.next_batch(FLAGS.batch_size)
    fd = {
        input_pl: batch_data,
        label_pl: batch_label
    }
    return fd


###########################
# model
###########################
def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _get_variable_with_regularization(name, shape, initializer, reg_w=net_config.regu_coef):
    var = tf.get_variable(
        name,
        shape,
        dtype=tf.float32,
        initializer=initializer,
        regularizer=tf.contrib.layers.l2_regularizer(reg_w)
    )
    return var






def get_inference_op(features):
    with tf.variable_scope("preprocess"):
        # split input into sequence feature and profile
        seq_features, profile = tf.split(features,
                                         [net_config.in_first_size, net_config.in_second_size],
                                         axis=2, name="split_features")
        embed_mat = _get_variable_with_regularization("embed_mat",
                                                      [net_config.in_first_size, net_config.embed_size],
                                                      tf.truncated_normal_initializer(stddev=0.1)
                                                      )
        seq_features_flat = tf.reshape(seq_features, [-1, net_config.in_first_size], "flatten")
        embed_feature = tf.reshape(
            tf.matmul(seq_features_flat, embed_mat, a_is_sparse=True, name="flat_embedding"),
            [-1, net_config.seq_len, net_config.embed_size],
            name="embedding"
        )

        preprocessed_feature = tf.concat([embed_feature, profile], 2)
        _activation_summary(preprocessed_feature)

    with tf.variable_scope("multiscal_conv"):
        # convolution with kernel 1
        kernel1 = _get_variable_with_regularization("kernel1",
                                                    net_config.kernel1,
                                                    tf.truncated_normal_initializer(stddev=0.1)
                                                    )
        bias1 = tf.get_variable("bias1", net_config.kernel1[-1], dtype=tf.float32,
                                initializer=tf.constant_initializer(value=0.5))
        z1 = tf.nn.conv1d(preprocessed_feature, kernel1, stride=1, padding="SAME", name="conv1")
        conv1 = tf.nn.relu(z1 + bias1, "relu1")

        # convolution with kernel 2
        kernel2 = _get_variable_with_regularization("kernel2",
                                                    net_config.kernel2,
                                                    tf.truncated_normal_initializer(stddev=0.1)
                                                    )
        bias2 = tf.get_variable("bias2", net_config.kernel2[-1], dtype=tf.float32,
                                initializer=tf.constant_initializer(value=0.5))
        z2 = tf.nn.conv1d(preprocessed_feature, kernel2, stride=1, padding="SAME", name="conv2")
        conv2 = tf.nn.relu(z2 + bias2, "relu2")

        # convolution with kernel3
        kernel3 = _get_variable_with_regularization("kernel3",
                                                    net_config.kernel3,
                                                    tf.truncated_normal_initializer(stddev=0.1)
                                                    )
        bias3 = tf.get_variable("bias3", net_config.kernel1[-1], dtype=tf.float32,
                                initializer=tf.constant_initializer(value=0.5))
        z3 = tf.nn.conv1d(preprocessed_feature, kernel3, stride=1, padding="SAME", name="conv3")
        conv3 = tf.nn.relu(z3 + bias3, "relu3")

        concat_conv = tf.concat([conv1, conv2, conv3], axis=2)
        _activation_summary(concat_conv)

    with tf.variable_scope("fully_connected1"):
        weight = _get_variable_with_regularization("weight",
                                                   net_config.fc1,
                                                   tf.truncated_normal_initializer(stddev=0.5))
        bias = tf.get_variable("bias", net_config.fc1[-1], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.5))

        flat_conv = tf.reshape(concat_conv, [-1, net_config.fc1[0]])
        hidden1 = tf.nn.relu(tf.matmul(flat_conv, weight) + bias, name="hidden")

    with tf.variable_scope("fully_connected2"):
        weight = _get_variable_with_regularization("weight",
                                                   net_config.fc2,
                                                   tf.truncated_normal_initializer(stddev=0.5))
        bias = tf.get_variable("bias", net_config.fc2[-1], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=0.5))

        logits = tf.nn.relu(tf.matmul(hidden1, weight) + bias, name="logits")


    return logits


def get_loss_op(logits, labels):
    # split the logits and labels into secondary structure and solvent  accessibility
    logits_ss, logits_sa = tf.split(logits,
                                    [net_config.label_first_size, net_config.label_second_size],
                                    axis=1, name="split_logits")

    flat_labels = tf.reshape(labels, [-1, net_config.label_size], name="flat_labels")
    labels_ss, labels_sa = tf.split(flat_labels,
                                    [net_config.label_first_size, net_config.label_second_size],
                                    axis=1, name="split_labels")

    # calculate the losses separately
    entropy_ss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_ss, logits=logits_ss, name="entropy_ss")
    entropy_sa = tf.nn.softmax_cross_entropy_with_logits(labels=labels_sa, logits=logits_sa, name="entropy_sa")
    # add regularization
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_mean(entropy_ss) + tf.reduce_mean(entropy_sa) + tf.add_n(reg_losses)
    return loss


def get_train_op(loss, global_step):
    opt = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
    return opt

##########################
# test
##########################


###########################
# training
###########################
def run_epoch(session, fetches, input_pl, label_pl, data_set):
    """
    run one epoch
    :param session: the session to run on
    :param fetches: ops to be run
    :param input_pl: input place holder
    :param label_pl: label place holder
    :param data_set: the data set to input
    :return: return the average values of fetches
    """
    ret = np.zeros(len(fetches))
    iter_num = 0
    while True:
        data, label, is_end = data_set.next_batch(FLAGS.batch_size)
        fd = {
            input_pl: data,
            label_pl: label
        }
        val = session.run(fetches, feed_dict=fd)
        #ret += np.asarray(val)
        iter_num += 1

        if iter_num % 20 == 0:
            print(val)

        if is_end:
            return val#(ret / iter_num).tolist()



def main():
    data = read_data_from_example("/Users/ming/projects/DeepLearning/data/cullpdb+profile_6133.npy")
    train_data, valid_data, test_data = get_train_valid_test(data, [0.7, 0.15, 0.15])
    train_dataset = DataSet(train_data[0], train_data[1])
    valid_dataset = DataSet(valid_data[0], valid_data[1])
    test_dataset = DataSet(test_data[0], test_data[1])

    #data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)

    gf = tf.Graph()
    with gf.as_default():
        input_pl = tf.placeholder(dtype=tf.float32,
                                  shape=[FLAGS.batch_size, net_config.seq_len, net_config.in_size],
                                  name="input_place_holder")
        label_pl = tf.placeholder(dtype=tf.float32,
                                  shape=[FLAGS.batch_size, net_config.seq_len, net_config.label_size],
                                  name="label_place_holder")
        global_step = tf.contrib.framework.get_or_create_global_step()
        with tf.name_scope("Train"):
            logits = get_inference_op(input_pl)
            train_loss = get_loss_op(logits, label_pl)
            train_op = get_train_op(train_loss, global_step)


        '''
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(FLAGS.epoch_num):
                if sv.should_stop():
                    break
                train_loss = run_epoch(session, [train_op], input_pl, label_pl, train_dataset)
        '''
    with tf.Session(graph=gf) as sess:
        tf.global_variables_initializer().run()
        for step in range(10000):
            data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)
            feed_dict = {
                input_pl: data,
                label_pl: label
            }
            _, loss_value = sess.run([train_op, train_loss], feed_dict=feed_dict)
            if step % 10 == 0:
                print(loss_value)

    return

if __name__ == "__main__":
    main()
