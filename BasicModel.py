import numpy as np
import tensorflow as tf
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






def get_inference_op(seq_features, profile):
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

    return logits_ss, logits_sa


def split_labels(labels):
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


def split_features(features):
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


def get_loss_op(logits_ss, logits_sa, labels_ss, labels_sa):
    with tf.variable_scope("loss_operator"):
        # calculate the losses separately
        entropy_ss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_ss, logits=logits_ss, name="entropy_ss")
        entropy_sa = tf.nn.softmax_cross_entropy_with_logits(labels=labels_sa, logits=logits_sa, name="entropy_sa")
        # add regularization
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.reduce_mean(entropy_ss) + tf.reduce_mean(entropy_sa) + tf.add_n(reg_losses)
        tf.summary.scalar("loss", loss)
    return loss


def get_train_op(loss, global_step):
    with tf.variable_scope("objective_funciton"):
        opt = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
        tf.summary.scalar('global_step', global_step)
    return opt

##########################
# test
##########################


def get_accuracy_op(logits_ss, one_hot_labels, input_length):
    """
    calculate the q_8 (8 classes accuracy) not the "Q8" accuracy for 3 classes
    :param logits:
    :param one_hot_labels:
    :return:
    """
    with tf.variable_scope("testing"):
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
        q_8 = tf.div(true_positive, total)

    return q_8

###########################
# training
###########################

