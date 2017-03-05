import tensorflow as tf
from data_process import *
from BasicModel import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_path', '/home/dm/PycharmProjects/Deep_protein/exp_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('epoch_num', 1000,
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


def evaluate(session, e_model):
    """

    :param session:
    :param e_model:
    :return:
    """
    e_val = 0
    example_count = 0
    for i in range(e_model.batch_per_epoch):
        ret = session.run(e_model.fetches)
        e_val += ret["evaluation"]
        example_count += ret["example_count"]
    else:
        e_val /= i
    print("evaluate %d proteins, get accuracy: %f" % (example_count/net_config.seq_len, e_val))


def main():
    #data = read_data_from_example("/home/dm/data_sets/cullpdb+profile_6133_filtered.npy")
    #train_data, valid_data, test_data = get_train_valid_test(data, [0.1, 0.15, 0])


    #data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)

    gf = tf.Graph()
    with gf.as_default():
        train_model = Model(FLAGS, "train", ["/home/dm/data_sets/cb513+profile_split1.tfrecords"])
        train_model.build_graph()
        summary_op = tf.summary.merge_all()
        ft = train_model.fetches
        ft["summary"] = summary_op

        sv = tf.train.Supervisor(logdir=FLAGS.save_path, summary_op=None, save_model_secs=300)
        with sv.managed_session() as sess:
            iter = 0
            #sv.start_queue_runners(sess)

            while True:#not sv.should_stop():
                try:
                    ret = sess.run(ft)

                except tf.errors.OutOfRangeError:
                    break
                iter += 1
                sv.summary_computed(sess, ret["summary"])
                print(ret["loss"], ret["evaluation"])
                '''
                if iter % 100 == 0:
                    # validation
                    evaluate(sess, valid_model)
                '''


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
