import tensorflow as tf
from BasicModel import *
from data_process import *
import pickle as pkl
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string("checkpoint_dir", "/home/dm/PycharmProjects/Deep_protein/result",
                           "the directory for checkpoint files")




def run_once(session, model, data_set):
    accuracy = 0
    count = 0
    while True:
        b_data, b_label, b_len, is_end = data_set.next_batch(FLAGS.batch_size)
        count += len(b_data)
        fd = model.get_fed_dict(input_data=b_data, input_label=b_label, input_len=b_len)
        val_dict = session.run(model.fetches, feed_dict=fd)
        accuracy += val_dict["evaluation"] * len(b_data)
        print(val_dict["evaluation"])
        if is_end:
            break
    return accuracy / count, count, val_dict


def evaluate():
    log_dir = "/home/dm/PycharmProjects/Deep_protein/exp_log"

    test_data_name = '/home/dm/data_sets/cb513+profile_split1.npy'
    data, label, length = read_data_from_example(test_data_name)
    dataset = DataSet(data, label, length)

    '''
    valid_file = '/home/dm/data_sets/cullpdb+profile_6133_filtered_valid.pkl'
    fh = open(valid_file, "rb")
    valid_data = pkl.load(fh)
    fh.close()
    dataset = DataSet(valid_data[0], valid_data[1], valid_data[2])
    '''

    gf = tf.Graph()
    with gf.as_default():
        test_model = Model(FLAGS, "test")
        test_model.build_graph()


    with tf.Session(graph=gf) as session:
        if FLAGS.using_moving_average:
            var_list = test_model.moving_average_maintainer.variables_to_restore()
        else:
            var_list = None
        saver = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(session, ckpt.model_checkpoint_path)
            acc, count, ret = run_once(session, test_model, dataset)
            print(acc, count)

if __name__ == "__main__":
    tf.app.flags.DEFINE_integer("batch_size", 64,
                                "number of batches")
    tf.app.flags.DEFINE_integer("using_moving_average", True,
                                "whether store from moving averages ")
    evaluate()
