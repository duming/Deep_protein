from BasicModel import Model, _activation_summary
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as wrapper
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import core as layers_core


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
        self.encoder_hidden_num = 50
        # decoder part
        self.decoder_cell_num = 1
        self.decoder_hidden_num = 50

        # output part
        self.is_predict_sa = False
        if self.is_predict_sa is True:
            self.output_size = 8
        else:
            self.output_size = 12

        # moving average
        self.decay_rate = 0.999

seq2seq_config = Seq2seq_net_config()

class Seq2seqModel(Model):
    def __init__(self, config, mode, input_file_list=None, net_config=seq2seq_config):
        super(Seq2seqModel, self).__init__(config, mode, input_file_list, net_config)
        self.embed_output = None
        self.conv_output = None
        self.encoder_output = None
        self.decoder_output = None

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
            _outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, _inputs,
                                                          sequence_length=seq_len,
                                                          dtype=tf.float32,
                                                          )
            # concatenate forward and backward tensor
            _rnn_output = tf.concat(_outputs, axis=2)
        self.encoder_output = _rnn_output
        return _rnn_output

    def build_decoder(self, encoder_output, seq_len):
        with tf.variable_scope("rnn_decoder", initializer=self.weight_initializer):
            # prepare helper
            if self.is_training:
                # TODO add "GO" signal before input
                helper = helper_py.TrainingHelper(encoder_output, seq_len)
            else:
                # TODO prepare embed matrix, start_token, end_token
                embed_mat = None
                start_tokens = None
                end_token = None
                helper = helper_py.GreedyEmbeddingHelper(embed_mat, start_tokens, end_token)

            cell = self.get_rnn_cell()
            _decoder = basic_decoder.BasicDecoder(
                cell=cell,
                helper=helper,
                initial_state=cell.zero_state(
                    dtype=dtypes.float32, batch_size=self.batch_size),
                output_layer=layers_core.Dense(self.net_config.ouput_size, use_bias=False))

            batch_max_len = tf.reduce_max(seq_len)
            final_outputs, final_state = decoder.dynamic_decode(
                _decoder, output_time_major=False,
                maximum_iterations=batch_max_len)
            self.decoder_output = final_outputs[0]



    def build_inference(self, seq_features, profile, is_reuse=False):
        self.build_embedding(seq_features, profile)
        self.build_convolution(self.conv_output)


    def build_loss(self, logits_ss, logits_sa, labels_ss, labels_sa, seq_len):
        pass

