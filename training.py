import tensorflow as tf
from data_process import *
from BasicModel import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_path', '/home/dm/PycharmProjects/Deep_protein/exp_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('epoch_num', 100,
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
    data = read_data_from_example("/home/dm/data_sets/cullpdb+profile_6133_filtered.npy")
    train_data, valid_data, test_data = get_train_valid_test(data, [0.1, 0.15, 0.15])
    train_dataset = DataSet(train_data[0], train_data[1], train_data[2])
    valid_dataset = DataSet(valid_data[0], valid_data[1], valid_data[2])
    test_dataset = DataSet(test_data[0], test_data[1], valid_data[2])

    #data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)

    gf = tf.Graph()
    with gf.as_default():
        with tf.variable_scope("inputs"):
            with tf.device("/cpu:0"):
                input_data = tf.constant(train_data[0], dtype=tf.float32)
                input_label = tf.constant(train_data[1], dtype=tf.int32)
                input_len = tf.constant(train_data[2], dtype=tf.int32)

            data, label, lengths = tf.train.slice_input_producer(
                [input_data, input_label, input_len], num_epochs=FLAGS.epoch_num)
            batch_data, batch_label, batch_lengths = tf.train.batch(
                [data, label, lengths], batch_size=1)#=FLAGS.batch_size)
            global_step = tf.contrib.framework.get_or_create_global_step()
            # split input
            seq_features, profile = split_features(batch_data)
            labels_ss, labels_sa = split_labels(batch_label)

        logits_ss, logits_sa = get_inference_op(seq_features, profile)
        train_loss = get_loss_op(logits_ss, logits_sa, labels_ss, labels_sa)
        train_op = get_train_op(train_loss, global_step)
        test_op = get_accuracy_op(logits_ss, labels_ss, batch_lengths)
        summary_op = tf.summary.merge_all()


        sv = tf.train.Supervisor(logdir=FLAGS.save_path, summary_op=None, save_model_secs=30)
        with sv.managed_session() as sess:
            iter = 0
            #sv.start_queue_runners(sess)
            while not sv.should_stop():
                ret = sess.run({"loss": train_loss,
                                "objective": train_op,
                                "summary": summary_op,
                                "accuracy": test_op,
                                }
                               )
                iter += 1
                print(iter, ret["loss"], ret["accuracy"])
                sv.summary_computed(sess, ret["summary"])


        '''
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())


    with tf.Session(graph=gf) as sess:
        #tf.global_variables_initializer().run()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        while not coord.should_stop():
            #data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)

            ret = sess.run({"loss": train_loss,
                            "objective": train_op,
                            "summary": summary_op}
                           )
            step += 1
            print(step, ret["loss"])

    '''
    return

if __name__ == "__main__":
    main()