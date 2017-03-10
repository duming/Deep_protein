import tensorflow as tf
from data_process import *
from BasicModel import *
import pickle as pkl
from testing import run_once


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_path', '/home/dm/PycharmProjects/Deep_protein/exp_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('epoch_num', 10000,
                            """Number of epoch to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer("batch_size", 128,
                            "number of batches")



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
    valid_file = "/home/dm/data_sets/cullpdb+profile_6133_filtered_valid.pkl"
    fh = open(valid_file, "rb")
    valid_data = pkl.load(fh)
    fh.close()
    valid_dataset = DataSet(valid_data[0], valid_data[1], valid_data[2])

    gf = tf.Graph()
    with gf.as_default():
        train_model = Model(FLAGS, "train", ["/home/dm/data_sets/cullpdb+profile_6133_filtered_train.tfrecords"])
        train_model.build_graph()
        ft = train_model.fetches

        with tf.name_scope("valid"):
            valid_model = Model(FLAGS, "valid")
            valid_model.build_graph()

        sv = tf.train.Supervisor(logdir=FLAGS.save_path, summary_op=None, save_model_secs=300)
        valid_writer = tf.summary.FileWriter(FLAGS.save_path + '/valid')

        with sv.managed_session() as sess:
            iter = 0
            #sv.start_queue_runners(sess)

            while True:#not sv.should_stop():
                iter += 1
                if iter % 50 == 0:
                    # validation
                    valid_precision, count, valid_ret = run_once(sess, valid_model, valid_dataset)
                    print(valid_precision, count)
                    # TODO fix bug: when restoring from checkpoint, the summary graph doesn't plot properly.
                    # maybe caused by separately saving training summary and validation summary
                    valid_writer.add_summary(valid_ret["summary"], global_step=valid_ret["step"])
                else:
                    try:
                        ret = sess.run(ft)

                    except tf.errors.OutOfRangeError:
                        break

                    sv.summary_computed(sess, ret["summary"])
                    print(ret["loss"], ret["evaluation"])





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
