import numpy as np
from data_process import *
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn import preprocessing, metrics
import tensorflow as tf
from BasicModel import *
import os
from training import EarlyStop

cf_CB513 = np.asarray(
    [[11679, 9, 2888, 216, 0, 974, 987, 1167],
     [588, 31, 294, 12, 0, 98, 66, 92],
     [2107, 8, 14919, 41, 0, 410, 201, 330],
     [635, 0, 172, 790, 0, 904, 80, 551],
     [4, 0, 2, 3, 0, 17, 0, 4],
     [678, 2, 216, 250, 0, 24394, 76, 541],
     [3077, 5, 888, 146, 0, 688, 2072, 1440],
     [1616, 1, 447, 385, 0, 1822, 520, 5222]]
)

cf_casp10 =np.asarray(
    [[3894, 4, 402, 42, 0, 113, 182, 163],
     [69, 3, 23, 3, 0, 10, 5, 3],
     [316, 2, 2855, 21, 0, 59, 64, 35],
     [118, 0, 12, 125, 0, 81, 29, 51],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [102, 0, 46, 48, 0, 3535, 9, 86],
     [392, 1, 98, 22, 0, 58, 314, 140],
     [239, 1, 59, 70, 0, 212, 95, 918]]
)

cf_casp11 = np.asarray(
    [[1921, 2, 359, 24, 0, 93, 134, 97],
     [50, 5, 30, 4, 0, 6, 8, 5],
     [288, 1, 2368, 16, 0, 35, 55, 35],
     [79, 0, 35, 89, 0, 60, 24, 41],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [115, 0, 68, 43, 0, 2851, 17, 57],
     [314, 0, 102, 23, 0, 73, 236, 131],
     [176, 0, 45, 49, 0, 190, 90, 581]
     ]
)

print(cf_CB513)
print(cf_CB513.sum())

#_, _, lengths = read_data_from_example("/home/dm/data_sets/cb513+profile_split1.npy")

#print(sum(lengths))

label_dict_8 = dict(enumerate(['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']))
# H=HGI, E=EB, C=STC
label_dict_3 = {
    0: 'H',
    1: 'E',
    2: 'C'
}

alphaIndex = range(3, 6)
betaIndex = range(1, 3)
coilIndex = list([0]) + list(range(6, 8))

def Q_accuracy(conf_mat):
    t_sum = conf_mat.diagonal().sum()
    all_sum = conf_mat.sum()
    return t_sum/all_sum


def conf8_to_conf3(conf_mat_8):
    # shrink rows
    alpha = conf_mat_8[alphaIndex, :].sum(axis=0)
    beta = conf_mat_8[betaIndex, :].sum(axis=0)
    coil = conf_mat_8[coilIndex, :].sum(axis=0)

    tmp = np.stack([alpha, beta, coil], axis=0)
    #
    alpha = tmp[:, alphaIndex].sum(axis=1)
    beta = tmp[:, betaIndex].sum(axis=1)
    coil = tmp[:, coilIndex].sum(axis=1)
    conf_mat_3 = np.stack([alpha, beta, coil], axis=1)
    return conf_mat_3


#print(conf8_to_conf3(cf_CB513))

#print(Q_accuracy(cf_CB513))
#print(Q_accuracy(conf8_to_conf3(cf_CB513)))

#print(Q_accuracy(cf_casp10))
#print(Q_accuracy(conf8_to_conf3(cf_casp10)))

#print(Q_accuracy(cf_casp11))
#print(Q_accuracy(conf8_to_conf3(cf_casp11)))

def test_accuracy_op():
    """
    test the correctness of q_8 accuracy operation
    :return:
    """
    def label_to_q8(labels, predictions):
        conf = metrics.confusion_matrix(labels, predictions)
        conf = conf[:8, :8]
        print(conf)
        return Q_accuracy(conf)

    true_label = [[i]*100 for i in range(9)]
    true_label = np.asarray(true_label).reshape([-1, 1])
    predict = true_label#[::-1]
    enc = preprocessing.OneHotEncoder(sparse=False)
    enc.fit(true_label)
    one_hot_y = enc.transform(true_label)
    one_hot_predict = enc.transform(predict)

    one_hot_y[:, -1] = np.zeros(one_hot_y.shape[0])
    one_hot_predict[:, -1] = np.zeros(one_hot_predict.shape[0])
    one_hot_predict = one_hot_predict * 0.9


    print(label_to_q8(true_label, predict))

    graph = tf.Graph()
    with graph.as_default():
        label_v = tf.Variable(one_hot_y, dtype=tf.int32)
        pred_v = tf.Variable(one_hot_predict, dtype=tf.float32)
        op = get_accuracy_op(pred_v, label_v)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        ret = session.run(op)
        print(ret)




#test_accuracy_op()

data6133_filtered = "/home/dm/data_sets/cullpdb+profile_6133_filtered.npy"
data6133_filtered_record = "/home/dm/data_sets/cullpdb+profile_6133_filtered.tfrecords"
#covert_ICML2014_to_record(data6133_filtered)
#split_and_convert(data6133_filtered)
#read_record_file_for_test(data6133_filtered_record)

"""
def test_input():
    graph = tf.Graph()
    file_name = "/home/dm/data_sets/cullpdb+profile_6133_filtered.tfrecords"
    with graph.as_default():
        filename_queue = tf.train.string_input_producer(
            [file_name], num_epochs=1000)
        op = batch_input(file_name, 1, 64)

        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    with tf.Session(graph=graph) as session:

        # Initialize the variables (the trained variables and the
        # epoch counter).
        session.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                ret = session.run(op)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done  %d steps.' % step)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)


"""

#test_input()
"""
f_name = "/home/dm/data_sets/cb513+profile_split1.npy"
data_513 = read_data_from_example(f_name)
n_name = os.path.splitext(f_name)[0] + ".tfrecords"
convert_ICML2014_to_record(data_513, n_name)
"""


#es = EarlyStop(3)
#updates = list(np.linspace(1, 4, 10)) + [4, 4, 5, 0, 0, 2, 2]

#for i in updates:
#    up = es.should_stop_update(i)
#    print(i, up)
print(tf.__version__)

def test_something():
    graph = tf.Graph()
    file_name = "/home/dm/data_sets/cullpdb+profile_6133_filtered.tfrecords"
    with graph.as_default():
        input_pl = tf.placeholder(tf.float32, (None, None, 3))
        _input = tf.convert_to_tensor(input_pl)
        _input_slice = tf.slice(_input, [0, 0, 0], [-1, 1, -1])
        _slice_shape = tf.shape(_input_slice)
        start_signal = tf.ones(_slice_shape)
        _input_c = tf.concat([start_signal, _input], axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        batch_size = tf.slice(tf.shape(_input), [0], [1])

    # Create a session for running operations in the Graph.
    with tf.Session(graph=graph) as session:

        # Initialize the variables (the trained variables and the
        # epoch counter).
        session.run(init_op)
        for i in range(1, 4):
            value = np.asarray([[[i]*3]*4]*i, dtype=np.float32)
            fd = {input_pl: value}
            ret = session.run(batch_size, feed_dict=fd)
            print(ret)


start_code = np.ones(8)
end_code = np.zeros(8)
embed_mat = np.identity(8)
embed_mat = np.vstack([start_code, end_code, embed_mat])
print(embed_mat)
test_something()
