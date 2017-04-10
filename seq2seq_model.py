from BasicModel import Model, _activation_summary
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as wrapper
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
import numpy as np

class Seq2seq_net_config(object):
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
        # encoder part
        self.encoder_cell_num = 1
        self.encoder_hidden_num = 128
        # decoder part
        self.decoder_cell_num = 1
        self.decoder_hidden_num = 128

        # output part
        self.is_predict_sa = False
        if self.is_predict_sa is False:
            self.output_size = 8
            self.sa_loss_ratio = 0
        else:
            self.output_size = 12
            self.sa_loss_ratio = 0.1

        # moving average
        self.decay_rate = 0.999

seq2seq_config = Seq2seq_net_config()


class Seq2seqModel(Model):
    def __init__(self, config, mode, input_file_list=None, net_config=seq2seq_config):
        super(Seq2seqModel, self).__init__(config, mode, input_file_list, net_config)
        self.embed_output = None
        self.conv_output = None
        self.encoder_output = None
        self.encoder_final_state = None
        self.decoder_output = None
        # max sequence length in current batch
        self.curr_max_len = None

    def split_labels(self, labels):
        """
        split the label to secondary structure part and solvent accessibility part
        override the function, which don't flatten the labels
        :param labels:
        :return:
        """
        with tf.variable_scope("split_label"):
            labels_ss, labels_sa = tf.split(labels,
                                            [self.net_config.label_first_size, self.net_config.label_second_size],
                                            axis=2, name="split_labels")

        return labels_ss, labels_sa

    def build_input(self):
        """
        override to adjust the shape of labels
        :return:
        """
        super(Seq2seqModel, self).build_input()
        # adjust the shape
        _max_len = tf.reduce_max(self.seq_lens)
        self.curr_max_len = _max_len
        self.labels_ss = tf.slice(self.labels_ss, [0, 0, 0], [-1, _max_len, -1])
        self.labels_sa = tf.slice(self.labels_sa, [0, 0, 0], [-1, _max_len, -1])
        # need to reshape the labels in order to set the last dimension to a fix size.
        # Since the the rnn cell need the input to have a fixed last dimension
        self.labels_ss = tf.reshape(self.labels_ss, [-1, _max_len, self.net_config.label_first_size])
        self.labels_sa = tf.reshape(self.labels_sa, [-1, _max_len, self.net_config.label_second_size])

    def build_embedding(self, seq_features, profile):
        with tf.variable_scope("preprocess"):
            embed_mat = self._get_variable_with_regularization("embed_mat",
                                                               [self.net_config.in_first_size,
                                                                self.net_config.embed_size]
                                                               )
            seq_features_flat = tf.reshape(seq_features, [-1, self.net_config.in_first_size], "flatten")
            embed_feature = tf.reshape(
                tf.matmul(seq_features_flat, embed_mat, a_is_sparse=True, name="flat_embedding"),
                [-1, self.net_config.seq_len, self.net_config.embed_size],
                name="embedding"
            )

            embed_output = tf.concat([embed_feature, profile], 2)
            _activation_summary(embed_output)
            self.embed_output = embed_output
            return embed_output

    def build_convolution(self, embed_output):
        with tf.variable_scope("multiscal_conv"):
            # convolution with kernel 1
            kernel1 = self._get_variable_with_regularization("kernel1",
                                                             self.net_config.kernel1
                                                             )
            bias1 = tf.get_variable("bias1", self.net_config.kernel1[-1], dtype=tf.float32,
                                    initializer=self.bias_initializer)
            z1 = tf.nn.conv1d(embed_output, kernel1, stride=1, padding="SAME", name="conv1")
            conv1 = z1 + bias1

            # convolution with kernel 2
            kernel2 = self._get_variable_with_regularization("kernel2",
                                                             self.net_config.kernel2
                                                             )
            bias2 = tf.get_variable("bias2", self.net_config.kernel2[-1], dtype=tf.float32,
                                    initializer=self.bias_initializer)
            z2 = tf.nn.conv1d(embed_output, kernel2, stride=1, padding="SAME", name="conv2")
            conv2 = z2 + bias2

            # convolution with kernel3
            kernel3 = self._get_variable_with_regularization("kernel3",
                                                             self.net_config.kernel3
                                                             )
            bias3 = tf.get_variable("bias3", self.net_config.kernel1[-1], dtype=tf.float32,
                                    initializer=self.bias_initializer)
            z3 = tf.nn.conv1d(embed_output, kernel3, stride=1, padding="SAME", name="conv3")
            conv3 = z3 + bias3

            concat_conv = tf.concat([conv1, conv2, conv3], axis=2)
            conv_relu = tf.nn.relu(concat_conv, name="relu")
            concat_conv = tf.contrib.layers.batch_norm(conv_relu, center=True, scale=True, decay=0.9,
                                                       is_training=self.is_training, scope="batch_norm")
            _activation_summary(concat_conv)
            self.conv_output = concat_conv
            return concat_conv

    def build_encoder(self, features, seq_len):
        with tf.variable_scope("rnn_encoder", initializer=self.weight_initializer):
            # construct multilayer rnn
            _inputs = features
            f_cell = self.get_rnn_cell(hidden_units=self.net_config.encoder_hidden_num, is_dropout=False)
            b_cell = self.get_rnn_cell(hidden_units=self.net_config.encoder_hidden_num, is_dropout=False)
            """
            _outputs, _states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, _inputs,
                                                          sequence_length=seq_len,
                                                          dtype=tf.float32,
                                                          )
            """
            _outputs, _states = tf.nn.dynamic_rnn(f_cell, _inputs, sequence_length=seq_len, dtype=tf.float32)

            # concatenate forward and backward tensor
            _rnn_output = tf.concat(_outputs, axis=2)
            _rnn_final_state = tf.concat(_states, axis=-1)
        self.encoder_output = _rnn_output
        self.encoder_final_state = _rnn_final_state
        return _rnn_output, _rnn_final_state

    @staticmethod
    def pad_start_signal(inputs):
        """
        pad the inputs with start signal of all ones
        :param inputs: tensor have shape [batch_size, seq_len, feature]
        :return: padded inputs [batch_size, 1 + seq_len, seq_len]
        """
        _input_slice = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
        _slice_shape = tf.shape(_input_slice)
        start_signal = tf.zeros(_slice_shape, dtype=tf.float32)
        _input_c = tf.concat([start_signal, inputs], axis=1)
        return _input_c

    def build_decoder(self, encoder_output, encoder_final_state, target, seq_len):
        """

        :param encoder_output: the output after each step of encoder execution, only useful for attention
        :param encoder_final_state: the final state of encoder, should be the initial state of decoder
                                    when the number of hidden units of encoder and decoder are not equal,  there should
                                    be a conversion
        :param target: target must be one-hot encoding
        :param seq_len:
        :return:
        """
        with tf.variable_scope("rnn_decoder", initializer=self.weight_initializer):
            # prepare helper
            if self.is_training:
                target = tf.cast(target, tf.float32)
                # add "GO" signal before decoder input
                # the decoder input should be the targets(labels)
                _decoder_input = self.pad_start_signal(target)
                helper = helper_py.TrainingHelper(_decoder_input, seq_len)
            else:
                # prepare embed matrix, start_token, end_token
                start_code = np.zeros(self.net_config.output_size)
                end_code = np.zeros(self.net_config.output_size)
                embed_mat = np.identity(self.net_config.output_size)
                embed_mat = np.vstack([embed_mat, start_code])
                embed_mat = tf.convert_to_tensor(embed_mat, dtype=tf.float32)
                # start_tokens vector of ones, length is batch size
                start_tokens = tf.ones(tf.slice(tf.shape(seq_len), [0], [1]), dtype=tf.int32) \
                    * (self.net_config.output_size + 1)
                # end token is a scalar
                end_token = 10 # set a impossible number that force decoding to the end
                helper = helper_py.GreedyEmbeddingHelper(embed_mat, start_tokens, end_token)

            # get rnn cell for decoder
            cell = self.get_rnn_cell(hidden_units=self.net_config.decoder_hidden_num)
            # build decoder
            _decoder = basic_decoder.BasicDecoder(
                cell=cell,
                helper=helper,
                #initial_state=cell.zero_state(
                #    dtype=dtypes.float32, batch_size=self.batch_size),
                initial_state=encoder_final_state,
                output_layer=layers_core.Dense(self.net_config.output_size, activation=tf.nn.relu,
                                               use_bias=False)
            )

            batch_max_len = tf.reduce_max(seq_len)
            # build the decoder layer
            final_outputs, final_state = decoder.dynamic_decode(
                _decoder, output_time_major=False,
                maximum_iterations=batch_max_len)
            self.decoder_output = final_outputs[0]
            return self.decoder_output

    def build_inference(self, seq_features, profile, is_reuse=False):
        """
        currently do not support multitask learning
        :param seq_features:
        :param profile:
        :param is_reuse:
        :return:
        """
        with tf.variable_scope("Model", reuse=is_reuse, initializer=self.weight_initializer):
            embed_output = self.build_embedding(seq_features, profile)
            conv_output = self.build_convolution(embed_output)
            encoder_output, encoder_final_state = self.build_encoder(conv_output, self.seq_lens)
            decoder_output = self.build_decoder(encoder_output, encoder_final_state, self.labels_ss, self.seq_lens)

            logits = decoder_output
            if self.net_config.is_predict_sa:
                logits_ss, logits_sa = tf.split(logits,
                                                [self.net_config.label_first_size, self.net_config.label_second_size],
                                                axis=1, name="split_logits")
            else:
                logits_ss = logits
                logits_sa = None
            self.logits_ss = logits_ss
            self.logits_sa = logits_sa

            if not self.is_training:
                # padding the output to the input length
                _curr_len = tf.shape(self.logits_ss)[1]
                self.logits_ss = tf.pad(self.logits_ss, [[0, 0], [0, self.curr_max_len - _curr_len], [0, 0]])

        return logits_ss, logits_sa

    def build_loss(self, logits_ss, logits_sa, labels_ss, labels_sa, seq_len):
        with tf.variable_scope("loss_operator"):
            # prepare weights
            weights = tf.cast(tf.sign(tf.reduce_max(labels_ss, axis=-1)), tf.float32)
            self.debug_outptu = weights
            labels_ss = tf.argmax(labels_ss, axis=-1)
            loss_ss = sequence_loss(
                logits_ss, labels_ss, weights,
                average_across_timesteps=True,
                average_across_batch=True)

            # add regularization
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = loss_ss + tf.add_n(reg_losses)
            if self.net_config.is_predict_sa:
                loss_sa = sequence_loss(
                    logits_sa, labels_sa, weights,
                    average_across_timesteps=True,
                    average_across_batch=True)
                loss_sa *= self.net_config.sa_loss_ratio
                loss += loss_sa

            tf.summary.scalar("loss", loss)
        self.loss = loss
        return loss

    def build_accuracy(self, logits_ss, one_hot_labels, input_length=None):
        """
        override to adapt the shape of tensor
        :param logits_ss:
        :param one_hot_labels:
        :param input_length:
        :return:
        """
        with tf.variable_scope("accuracy"):
            logits_ss = tf.reshape(logits_ss, [-1, self.net_config.output_size])
            one_hot_labels = tf.reshape(one_hot_labels, [-1, self.net_config.output_size])
            super(Seq2seqModel, self).build_accuracy(logits_ss, one_hot_labels)
