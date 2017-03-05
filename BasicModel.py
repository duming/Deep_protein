import numpy as np
import tensorflow as tf
from data_process import *
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
        self.label_first_size = 8
        self.label_second_size = 4

        self.label_size = self.label_first_size + self.label_second_size
        # preprocess
        self.in_first_size = 21
        self.in_second_size = 21
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





###########################
# model
###########################
class Model(object):
    def __init__(self, config, mode, input_file_list=None):
        assert mode in ["train", "eval", "inference"]
        self.epoch_num = config.epoch_num
        self.batch_size = config.batch_size
        self.mode = mode
        self.input_file_list = input_file_list
        # use for feed dict input
        self.batch_data_pl = None
        self.batch_label_pl = None

        self.global_step = None
        # split input
        self.seq_features = self.profile = None
        self.labels_ss = self.labels_sa = None

        self.logits_ss = None
        self.logits_sa = None
        self.loss = None
        self.train_op = None
        self.q_8_accuracy = None
        self.fetches = None
        self.filename_queue = None

    def get_fed_dict(self, file_list=None, input_data=None, input_label=None):
        """
        return the feed dict for run the graph
        :param file_list:
        :param input_data:
        :param input_label:
        :return:
        """
        if self.mode == "eval":
            # only need input file names, the graph will read the data
            # using preload data and feed the data and labels
            fd = {
                self.batch_data_pl: input_data,
                self.batch_label_pl: input_label
            }
        else:
            fd = None
        return fd

    def build_graph(self):
        self.build_input()
        self.build_inference(self.seq_features, self.profile)
        if self.mode == "train":
            self.loss = self.build_loss(self.logits_ss, self.logits_sa, self.labels_ss, self.labels_sa)
            self.train_op = self.build_train_op(self.loss, self.global_step)
            self.build_accuracy(self.logits_ss, self.labels_ss)
            self.fetches = {
                "loss": self.loss,
                "objective": self.train_op,
                "evaluation": self.q_8_accuracy
            }
        elif self.mode == "eval":
            self.loss = self.build_loss(self.logits_ss, self.logits_sa, self.labels_ss, self.labels_sa)
            self.build_accuracy(self.logits_ss, self.labels_ss)
            self.fetches = {
                "loss": self.loss,
                "evaluation": self.q_8_accuracy
            }
        elif self.mode == "inference":
            self.fetches = {
                "logits": self.logits_ss
            }

    ######################################################
    # input part
    ######################################################
    def build_input(self):
        with tf.variable_scope("inputs"):
            if self.mode == "train":
                # using queue input when training
                batch_data, batch_label, batch_lengths = \
                    self._batch_input(self.input_file_list, self.epoch_num, self.batch_size)

            else:
                # if is not training use feed dict instead
                batch_data = self.batch_data_pl = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, net_config.seq_len, net_config.in_size],
                                                              name="input_data_pl")
                batch_label = self.batch_label_pl = tf.placeholder(dtype=tf.int32,
                                                                   shape=[None, net_config.seq_len, net_config.in_size],
                                                                   name="input_label_pl")

            self.global_step = tf.contrib.framework.get_or_create_global_step()
            # split input
            self.seq_features, self.profile = self.split_features(batch_data)
            self.labels_ss, self.labels_sa = self.split_labels(batch_label)

    @staticmethod
    def _read_parse_records(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'label': tf.FixedLenFeature([DATA_SEQUENCE_LEN * 12], tf.int64),
                'data': tf.FixedLenFeature([DATA_SEQUENCE_LEN * 42], tf.float32),
                'length': tf.FixedLenFeature([1], tf.int64)
            })
        data = tf.reshape(tf.cast(features['data'], tf.float32), [DATA_SEQUENCE_LEN, -1])
        label = tf.reshape(tf.cast(features['label'], tf.int32), [DATA_SEQUENCE_LEN, -1])
        length = tf.cast(features['length'], tf.int32)
        return data, label, length

    def _batch_input(self, file_list, num_epochs, batch_size):
        with tf.name_scope('batch_input'):
            file_list_tensor = tf.Variable(file_list, dtype=tf.string)
            filename_queue = tf.train.string_input_producer(
                file_list_tensor, num_epochs=num_epochs)
            data, label, length = self._read_parse_records(filename_queue)
            b_data, b_label, b_length = tf.train.batch([data, label, length],
                                                       batch_size=batch_size,
                                                       capacity=128,
                                                       )
            self.filename_queue = filename_queue
        return b_data, b_label, b_length

    def split_labels(self, labels):
        """
        split the label to secondary structure part and solvent accessibility part
        :param labels:
        :return:
        """
        with tf.variable_scope("split_label"):
            flat_labels = tf.reshape(labels, [-1, net_config.label_size], name="flat_labels")
            labels_ss, labels_sa = tf.split(flat_labels,
                                            [net_config.label_first_size, net_config.label_second_size],
                                            axis=1, name="split_labels")

        return labels_ss, labels_sa

    def split_features(self, features):
        """
        split features into sequence part and profile part
        :param features:
        :return:
        """
        # split input into sequence feature and profile
        with tf.variable_scope("split_input"):
            seq_features, profile = tf.split(features,
                                             [net_config.in_first_size, net_config.in_second_size],
                                             axis=2, name="split_features")
        return seq_features, profile



    #################################################
    # Main model part
    #################################################
    def build_inference(self, seq_features, profile):
        with tf.name_scope("Model"):
            with tf.variable_scope("preprocess"):
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

                logits_ss, logits_sa = tf.split(logits,
                                                [net_config.label_first_size, net_config.label_second_size],
                                                axis=1, name="split_logits")

        self.logits_ss = logits_ss
        self.logits_sa = logits_sa
        return logits_ss, logits_sa

    ###################################################
    # Training loss part
    ###################################################
    def build_loss(self, logits_ss, logits_sa, labels_ss, labels_sa):
        with tf.variable_scope("loss_operator"):
            # calculate the losses separately
            entropy_ss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_ss, logits=logits_ss, name="entropy_ss")
            entropy_sa = tf.nn.softmax_cross_entropy_with_logits(labels=labels_sa, logits=logits_sa, name="entropy_sa")
            # add regularization
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.reduce_mean(entropy_ss) + tf.reduce_mean(entropy_sa) + tf.add_n(reg_losses)
            tf.summary.scalar("loss", loss)
        self.loss = loss
        return loss

    def build_train_op(self, loss, global_step):
        with tf.variable_scope("objective_funciton"):
            opt = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
            tf.summary.scalar('global_step', global_step)
        self.train_op = opt
        return opt

    ##########################
    # test
    ##########################


    def build_accuracy(self, logits_ss, one_hot_labels, input_length=None):
        """
        calculate the q_8 (8 classes accuracy) not the "Q8" accuracy for 3 classes
        :param logits:
        :param one_hot_labels:
        :return:
        """
        with tf.variable_scope("accuracy"):
            logits_ss = tf.nn.softmax(logits_ss)
            logits_preds = tf.add(tf.cast(tf.argmax(logits_ss, axis=1), dtype=tf.int32),
                                  tf.cast(
                                      tf.round(tf.reduce_sum(logits_ss, axis=1)),
                                      dtype=tf.int32)
                                  )

            true_labels = tf.add(tf.cast(tf.argmax(one_hot_labels, axis=1), dtype=tf.int32),
                                 tf.reduce_sum(one_hot_labels, axis=1))
            conf_mat = tf.confusion_matrix(logits_preds, true_labels, num_classes=9)
            conf_mat_8 = tf.slice(conf_mat, begin=[1, 1], size=[8, 8], name="confusion_mat8")

            tps = tf.diag_part(conf_mat_8, name="true_positives")
            true_positive = tf.cast(tf.reduce_sum(tps), tf.float32)
            total = tf.cast(tf.reduce_sum(conf_mat_8), tf.float32)
            q_8 = tf.div(true_positive, total, "accuracy_q8")

            # additional info for debug and monitor
            example_count = tf.slice(tf.shape(logits_ss), [0], [1])
            tf.summary.scalar("accuracy", q_8)
        self.q_8_accuracy = q_8
        return q_8, example_count








