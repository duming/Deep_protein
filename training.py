import tensorflow as tf
from data_process import *
from BasicModel import *


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
    avg_loss = 0
    iter_num = 0
    while True:
        data, label, lengths, is_end = data_set.next_batch(FLAGS.batch_size)
        fd = {
            input_pl: data,
            label_pl: label
        }
        val = session.run(fetches, feed_dict=fd)
        avg_loss += val["loss"]
        iter_num += 1
        if iter_num % 20 == 0:
            print(avg_loss/iter_num)

        if is_end:
            return val


def evaluate(session, eval_op, input_pl, label_pl, data_set):
    """

    :param session:
    :param eval_op:
    :param input_pl:
    :param label_pl:
    :param data_set:
    :return:
    """
    pass


def main():
    data = read_data_from_example("/Users/ming/projects/DeepLearning/data/cullpdb+profile_6133.npy")
    train_data, valid_data, test_data = get_train_valid_test(data, [0.7, 0.15, 0.15])
    train_dataset = DataSet(train_data[0], train_data[1], train_data[2])
    valid_dataset = DataSet(valid_data[0], valid_data[1], valid_data[2])
    test_dataset = DataSet(test_data[0], test_data[1], valid_data[2])

    #data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)

    gf = tf.Graph()
    with gf.as_default():
        with tf.variable_scope("inputs"):
            input_pl = tf.placeholder(dtype=tf.float32,
                                      shape=[None, net_config.seq_len, net_config.in_size],
                                      name="input_place_holder")
            label_pl = tf.placeholder(dtype=tf.float32,
                                      shape=[None, net_config.seq_len, net_config.label_size],
                                      name="label_place_holder")
            global_step = tf.contrib.framework.get_or_create_global_step()
            # split input
            seq_features, profile = split_features(input_pl)
            labels_ss, labels_sa = split_labels(label_pl)

        logits_ss, logits_sa = get_inference_op(seq_features, profile)
        train_loss = get_loss_op(logits_ss, logits_sa, labels_ss, labels_sa)
        train_op = get_train_op(train_loss, global_step)
        test_op = get_accuracy_op(logits_ss, labels_ss)
        summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(logdir=FLAGS.save_path, summary_op=None, save_model_secs=30)
        with sv.managed_session() as sess:
            for i in range(FLAGS.epoch_num):
                if sv.should_stop():
                    break
                ret = run_epoch(sess, {
                    "loss": train_loss,
                    "train_op": train_op,
                    "summary_op": summary_op
                    },
                    input_pl, label_pl, train_dataset
                )
                sv.summary_computed(sess, ret["summary_op"])
    '''
    with tf.Session(graph=gf) as sess:
        tf.global_variables_initializer().run()
        for step in range(10000):
            data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)
            feed_dict = {
                input_pl: data,
                label_pl: label
            }
            #_, loss_value = sess.run([train_op, train_loss], feed_dict=feed_dict)
            ret = run_epoch(sess, [train_loss, train_op], input_pl, label_pl, train_dataset)
            #if step % 10 == 0:
            #    print(loss_value)
    '''

    return

if __name__ == "__main__":
    main()