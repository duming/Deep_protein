import tensorflow as tf
from data_process import *
from BasicModel import *
import pickle as pkl
from testing import run_once
from operator import gt


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('save_path', '/home/dm/PycharmProjects/Deep_protein/exp_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('result_path', '/home/dm/PycharmProjects/Deep_protein/result',
                           """Directory where to write result model """
                          )

tf.app.flags.DEFINE_integer('epoch_num', 10000,
                            """Number of epoch to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer("batch_size", 64,
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


class EarlyStop(object):

    def __init__(self, max_step, metric=gt):
        """
        helper function for early stopping
        :param max_step: max step of none improvement, otherwise early stop
        :param is_stop_func: the function that compare new observation and current best
        """
        self.current_best = 0
        self.step_from_best = 0
        self.max_step = max_step
        self.metric = metric

    def should_stop_update(self, new_observation):
        if self.metric(new_observation, self.current_best):
            # new is better
            self.current_best = new_observation
            self.step_from_best = 0
            return [False, True]
        else:
            self.step_from_best += 1
            if self.step_from_best > self.max_step:
                return [True, False]
            else:
                return [False, False]


def main():
    valid_file = "/home/dm/data_sets/cullpdb+profile_6133_filtered_valid.pkl"
    fh = open(valid_file, "rb")
    valid_data = pkl.load(fh)
    fh.close()
    valid_dataset = DataSet(valid_data[0], valid_data[1], valid_data[2])
    early_stop = EarlyStop(20)
    gf = tf.Graph()
    with gf.as_default():
        train_model = Model(FLAGS, "train", ["/home/dm/data_sets/cullpdb+profile_6133_filtered_train.tfrecords"])
        train_model.build_graph()
        ft = train_model.fetches

        with tf.name_scope("valid"):
            valid_model = Model(FLAGS, "valid")
            valid_model.build_graph()

        valid_saver = tf.train.Saver(max_to_keep=1, name="valid_saver")
        train_saver = tf.train.Saver(max_to_keep=1, name="train_saver")
        sv = tf.train.Supervisor(logdir=FLAGS.save_path,saver=train_saver, summary_op=None, save_model_secs=300)
        with sv.managed_session() as sess:
            iter = 0
            while True:#not sv.should_stop():
                iter += 1
                if iter % 100 == 0:
                    # validation
                    valid_precision, count, valid_ret = run_once(sess, valid_model, valid_dataset)
                    print(valid_precision, count)

                    should_stop, should_save = early_stop.should_stop_update(valid_precision)
                    if should_save:
                        print("save current best with accuracy:%f" % valid_precision)
                        valid_saver.save(sess, FLAGS.result_path + "/best_model")
                    if should_stop:
                        print("early stop with best:%f" % early_stop.current_best)
                        break

                    # maybe caused by separately saving training summary and validation summary
                    sv.summary_computed(sess, valid_ret["summary"])
                    #valid_writer.add_summary(valid_ret["summary"], global_step=valid_ret["step"])
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
    # make directories
    if not tf.gfile.Exists(FLAGS.save_path):
        tf.gfile.MakeDirs(FLAGS.save_path)
    if not tf.gfile.Exists(FLAGS.result_path):
        tf.gfile.MakeDirs(FLAGS.result_path)

    main()
