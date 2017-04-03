import numpy as np
import tensorflow as tf
from data_process import *
########################
# file input output
########################


class NetConfig(object):
    def __init__(self):
        self.is_rnn = True
        self.is_dynamic_rnn = True
        # regularization coefficient
        self.regu_coef = 0.001
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

        # multiscal convolution
        self.kernel1 = [3, self.embed_size + self.in_second_size, 64]
        self.kernel2 = [7, self.embed_size + self.in_second_size, 64]
        self.kernel3 = [11, self.embed_size + self.in_second_size, 64]

        # rnn part
        self.unit_num = 128
        self.rnn_layer_num = 2
        self.rnn_dropout_prob = 0.5
        self.rnn_output_size = self.unit_num * 2 + self.kernel1[-1] + self.kernel2[-1] + self.kernel3[-1]

        # moving average
        self.decay_rate = 0.999

        # fully connected
        if self.is_rnn:
            self.fc1 = [self.rnn_output_size, int(self.rnn_output_size / 2)]
            self.fc2 = [self.fc1[-1], self.label_size]
        else:
            self.fc1 = [self.kernel1[-1] + self.kernel2[-1] + self.kernel3[-1],
                        32]
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


def _get_variable_with_regularization(name, shape, initializer=None, reg_w=net_config.regu_coef):
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
        assert mode in ["train", "test", "valid", "inference"]
        if mode == "train":
            self.epoch_num = config.epoch_num
            self.is_training = True
        else:
            self.is_training = False
        #self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        self.bias_initializer = tf.zeros_initializer()

        self.batch_size = config.batch_size
        self.mode = mode
        self.input_file_list = input_file_list
        # use for feed dict input
        self.batch_data_pl = None
        self.batch_label_pl = None
        self.seq_lens_pl = None

        self.global_step = None
        # sequence lengths for all examples in one batch
        self.seq_lens = None
        # split input
        self.seq_features = self.profile = None
        self.labels_ss = self.labels_sa = None

        self.logits_ss = None
        self.logits_sa = None
        self.loss = None
        self.train_op = None
        self.moving_average_train_op = None
        self.moving_average_maintainer = None
        self.q_8_accuracy = None
        self.fetches = None
        self.filename_queue = None
        self.rnn_output = None
        self.summary_loss = None
        self.summary_accuracy = None

    def get_fed_dict(self, file_list=None, input_data=None, input_label=None, input_len=None):
        """
        return the feed dict for run the graph
        :param file_list:
        :param input_data:
        :param input_label:
        :return:
        """
        if self.mode == "test" or "valid":
            # only need input file names, the graph will read the data
            # using preload data and feed the data and labels
            fd = {
                self.batch_data_pl: input_data,
                self.batch_label_pl: input_label,
                self.seq_lens_pl: input_len
            }
        else:
            fd = None
        return fd

    def build_graph(self):
        self.build_input()
        if self.mode == "train":
            self.build_inference(self.seq_features, self.profile, is_reuse=False)
            self.loss = self.build_loss(self.logits_ss, self.logits_sa, self.labels_ss, self.labels_sa, self.seq_lens)
            self.train_op = self.build_train_op(self.loss, self.global_step)
            self.build_moving_average()
            self.build_accuracy(self.logits_ss, self.labels_ss)
            summary_op = tf.summary.merge_all()


            self.fetches = {
                "loss": self.loss,
                "objective": self.moving_average_train_op,
                "evaluation": self.q_8_accuracy,
                "summary": summary_op
            }
        elif self.mode == "valid" or self.mode == "test":
            if self.mode == "valid":
                self.build_inference(self.seq_features, self.profile, is_reuse=True)
            else:
                self.build_inference(self.seq_features, self.profile, is_reuse=False)
            self.loss = self.build_loss(self.logits_ss, self.logits_sa, self.labels_ss, self.labels_sa, self.seq_lens)
            # build moving average only for restore
            if self.mode == "test":
                self.build_moving_average()

            self.build_accuracy(self.logits_ss, self.labels_ss)
            self.fetches = {
                "loss": self.loss,
                "evaluation": self.q_8_accuracy,
                "summary": tf.summary.merge([tf.summary.scalar("valid_loss", self.loss),
                                             tf.summary.scalar("valid_accuracy", self.q_8_accuracy)],
                                            name="valid_summary"),
                "step": self.global_step
            }
        elif self.mode == "inference":
            self.build_inference(self.seq_features, self.profile, is_reuse=False)
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
                batch_label = self.batch_label_pl = \
                    tf.placeholder(dtype=tf.int32,
                                   shape=[None, net_config.seq_len, net_config.label_size],
                                   name="input_label_pl")

                batch_lengths = self.seq_lens_pl = \
                    tf.placeholder(dtype=tf.int32, shape=[None], name="lengths_placeholder")
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            # split input
            self.seq_features, self.profile = self.split_features(batch_data)
            self.labels_ss, self.labels_sa = self.split_labels(batch_label)
            self.seq_lens = batch_lengths

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
                'length': tf.FixedLenFeature(1, tf.int64)
            })
        data = tf.reshape(tf.cast(features['data'], tf.float32), [DATA_SEQUENCE_LEN, -1])
        label = tf.reshape(tf.cast(features['label'], tf.int32), [DATA_SEQUENCE_LEN, -1])
        length = tf.cast(features['length'], tf.int32)
        return data, label, length

    def _batch_input(self, file_list, num_epochs, batch_size):
        with tf.name_scope('batch_input'):
            file_list_tensor = tf.Variable(file_list, dtype=tf.string, trainable=False)
            filename_queue = tf.train.string_input_producer(
                file_list_tensor, num_epochs=num_epochs)
            data, label, length = self._read_parse_records(filename_queue)
            b_data, b_label, b_length = tf.train.batch([data, label, length],
                                                       batch_size=batch_size,
                                                       capacity=128,
                                                       )
            b_length = tf.reshape(b_length, [-1])
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

    ####################################
    # rnn ops
    ####################################
    def get_rnn_cell(self):
        cell = tf.contrib.rnn.GRUCell(net_config.unit_num)

        if self.mode == "train":
            # apply dropout while training
            return tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=net_config.rnn_dropout_prob)
        else:
            return cell

    def build_rnn_layers(self, inputs, seq_len):
        """
        build the fully unrolled static rnn layer
        :param inputs:
        :param seq_len:
        :return:
        """
        with tf.variable_scope("rnn", initializer=self.weight_initializer):
            # convert tensor to list of tensors for rnn
            _inputs = tf.unstack(inputs, axis=1, name="unstack_for_rnn")
            # construct multilayer rnn
            for i in range(net_config.rnn_layer_num):
                with tf.name_scope("layer_%d"%i):
                    f_cell = self.get_rnn_cell()
                    b_cell = self.get_rnn_cell()
                    _inputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(f_cell, b_cell, _inputs,
                                                                            sequence_length=seq_len,
                                                                            dtype=tf.float32,
                                                                            scope="bidirectional_rnn_%d" % i
                                                                            )
            # convert the output of rnn back to tensor
            _rnn_output = tf.stack(_inputs, axis=1, name="stack_after_rnn")
            # concatenate the input of the first layer to the output of the last layer
            output = tf.concat([inputs, _rnn_output], axis=2, name="concat_input_output")
        self.rnn_output = output
        return output

    def build_dynamic_rnn_layers(self, inputs, seq_len):
        """
        build dynamic rnn layer
        :param inputs:
        :param seq_len:
        :return:
        """
        with tf.variable_scope("dynamic_rnn", initializer=self.weight_initializer):
            # construct multilayer rnn
            _inputs = inputs
            for i in range(net_config.rnn_layer_num):
                with tf.name_scope("layer_%d"%i):
                    f_cell = self.get_rnn_cell()
                    b_cell = self.get_rnn_cell()
                    _outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, _inputs,
                                                                  sequence_length=seq_len,
                                                                  dtype=tf.float32,
                                                                  scope="bidirectional_rnn_%d" % i
                                                                  )
                    # concatenate forward and backward tensor
                    _inputs = tf.concat(_outputs, axis=2)
            _rnn_output = _inputs
            # concatenate the input of the first layer to the output of the last layer
            output = tf.concat([inputs, _rnn_output], axis=2, name="concat_input_output")
        self.rnn_output = output
        return output

    #################################################
    # Main model part
    #################################################

    def build_inference(self, seq_features, profile, is_reuse=False):
        with tf.variable_scope("Model", reuse=is_reuse, initializer=self.weight_initializer):
            with tf.variable_scope("preprocess"):
                embed_mat = _get_variable_with_regularization("embed_mat",
                                                              [net_config.in_first_size, net_config.embed_size]
                                                              #tf.truncated_normal_initializer(stddev=0.01)
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
                                                            net_config.kernel1
                                                            #tf.truncated_normal_initializer(stddev=0.01)
                                                            )
                bias1 = tf.get_variable("bias1", net_config.kernel1[-1], dtype=tf.float32,
                                        initializer=self.bias_initializer)
                z1 = tf.nn.conv1d(preprocessed_feature, kernel1, stride=1, padding="SAME", name="conv1")
                conv1 = z1 + bias1

                # convolution with kernel 2
                kernel2 = _get_variable_with_regularization("kernel2",
                                                            net_config.kernel2
                                                            #tf.truncated_normal_initializer(stddev=0.01)
                                                            )
                bias2 = tf.get_variable("bias2", net_config.kernel2[-1], dtype=tf.float32,
                                        initializer=self.bias_initializer)
                z2 = tf.nn.conv1d(preprocessed_feature, kernel2, stride=1, padding="SAME", name="conv2")
                conv2 = z2 + bias2

                # convolution with kernel3
                kernel3 = _get_variable_with_regularization("kernel3",
                                                            net_config.kernel3
                                                            #tf.truncated_normal_initializer(stddev=0.01)
                                                            )
                bias3 = tf.get_variable("bias3", net_config.kernel1[-1], dtype=tf.float32,
                                        initializer=self.bias_initializer)
                z3 = tf.nn.conv1d(preprocessed_feature, kernel3, stride=1, padding="SAME", name="conv3")
                conv3 = z3 + bias3

                concat_conv = tf.concat([conv1, conv2, conv3], axis=2)
                conv_relu = tf.nn.relu(concat_conv, name="relu")
                concat_conv = tf.contrib.layers.batch_norm(conv_relu, center=True, scale=True, decay=0.9,
                                                           is_training=self.is_training, scope="batch_norm")
                _activation_summary(concat_conv)

            # rnn part
            if net_config.is_rnn:
                if net_config.is_dynamic_rnn:
                    rnn_output = self.build_dynamic_rnn_layers(concat_conv, self.seq_lens)
                else:
                    rnn_output = self.build_rnn_layers(concat_conv, self.seq_lens)
                rnn_output = tf.contrib.layers.batch_norm(rnn_output, center=True, scale=True, decay=0.9,
                                                          is_training=self.is_training, scope="rnn_batch_norm")
                _activation_summary(rnn_output)
                fc_input = rnn_output
            else:
                fc_input = concat_conv

            with tf.variable_scope("fully_connected1"):
                weight = _get_variable_with_regularization("weight",
                                                           net_config.fc1)
                bias = tf.get_variable("bias", net_config.fc1[-1], dtype=tf.float32,
                                       initializer=self.bias_initializer)

                flat_conv = tf.reshape(fc_input, [-1, net_config.fc1[0]])
                z1 = tf.matmul(flat_conv, weight) + bias
                norm_z1 = tf.contrib.layers.batch_norm(z1, center=True, scale=True,
                                                      is_training=self.is_training)
                hidden1 = tf.nn.relu(norm_z1, name="hidden")

            with tf.variable_scope("fully_connected2"):
                weight = _get_variable_with_regularization("weight",
                                                           net_config.fc2)
                bias = tf.get_variable("bias", net_config.fc2[-1], dtype=tf.float32,
                                       initializer=self.bias_initializer)
                z2 = tf.matmul(hidden1, weight) + bias
                #norm_z2 = tf.contrib.layers.batch_norm(z2, center=True, scale=True,
                #                                       is_training=self.is_training)
                logits = tf.nn.relu(z2, name="logits")

                logits_ss, logits_sa = tf.split(logits,
                                                [net_config.label_first_size, net_config.label_second_size],
                                                axis=1, name="split_logits")

        self.logits_ss = logits_ss
        self.logits_sa = logits_sa
        return logits_ss, logits_sa

    ###################################################
    # Training loss part
    ###################################################
    def build_loss(self, logits_ss, logits_sa, labels_ss, labels_sa, seq_len):
        with tf.variable_scope("loss_operator"):
            # calculate the losses separately
            entropy_ss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_ss, logits=logits_ss, name="entropy_ss")
            entropy_sa = tf.nn.softmax_cross_entropy_with_logits(labels=labels_sa, logits=logits_sa, name="entropy_sa")
            # for variable length input there should have a mask operation that zero out all output after sequence
            # length for each example, however here I skip this. Since all the labels after are already set to zero, and
            # the cross entropy should be zero when the label is zero
            total_len = tf.cast(tf.reduce_sum(seq_len), tf.float32)
            # add regularization
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.reduce_sum(entropy_ss) / total_len + tf.reduce_mean(entropy_sa) / total_len + tf.add_n(reg_losses)
            tf.summary.scalar("loss", loss)
        self.loss = loss
        return loss

    def build_train_op(self, loss, global_step):
        with tf.variable_scope("objective_funciton"):
            opt = tf.train.AdadeltaOptimizer(learning_rate=1).minimize(loss, global_step=global_step)
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

    ##################################
    # moving average
    ###################################
    def build_moving_average(self, decay=0.999):
        """
        Add operation that maintain moving averages of every trainable variables.
        And return a encapsulate train_op when the mode is training
        :param decay:
        :return:
        """
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        maintain_averages_op = ema.apply(var_list=var_list)
        self.moving_average_maintainer = ema
        if self.mode == "train":
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                with tf.control_dependencies([self.train_op]):
                    _train_op = tf.group(maintain_averages_op)

            self.moving_average_train_op = _train_op
            return _train_op




