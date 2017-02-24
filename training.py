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
    ret = np.zeros(len(fetches))
    iter_num = 0
    while True:
        data, label, is_end = data_set.next_batch(FLAGS.batch_size)
        fd = {
            input_pl: data,
            label_pl: label
        }
        val = session.run(fetches, feed_dict=fd)
        #ret += np.asarray(val)
        iter_num += 1

        if iter_num % 20 == 0:
            print(val)

        if is_end:
            return val#(ret / iter_num).tolist()



def main():
    data = read_data_from_example("/Users/ming/projects/DeepLearning/data/cullpdb+profile_6133.npy")
    train_data, valid_data, test_data = get_train_valid_test(data, [0.7, 0.15, 0.15])
    train_dataset = DataSet(train_data[0], train_data[1])
    valid_dataset = DataSet(valid_data[0], valid_data[1])
    test_dataset = DataSet(test_data[0], test_data[1])

    #data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)

    gf = tf.Graph()
    with gf.as_default():
        input_pl = tf.placeholder(dtype=tf.float32,
                                  shape=[FLAGS.batch_size, net_config.seq_len, net_config.in_size],
                                  name="input_place_holder")
        label_pl = tf.placeholder(dtype=tf.float32,
                                  shape=[FLAGS.batch_size, net_config.seq_len, net_config.label_size],
                                  name="label_place_holder")
        global_step = tf.contrib.framework.get_or_create_global_step()
        with tf.name_scope("Train"):
            logits = get_inference_op(input_pl)
            train_loss = get_loss_op(logits, label_pl)
            train_op = get_train_op(train_loss, global_step)


        '''
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(FLAGS.epoch_num):
                if sv.should_stop():
                    break
                train_loss = run_epoch(session, [train_op], input_pl, label_pl, train_dataset)
        '''
    with tf.Session(graph=gf) as sess:
        tf.global_variables_initializer().run()
        for step in range(10000):
            data, label, is_end = train_dataset.next_batch(FLAGS.batch_size)
            feed_dict = {
                input_pl: data,
                label_pl: label
            }
            _, loss_value = sess.run([train_op, train_loss], feed_dict=feed_dict)
            if step % 10 == 0:
                print(loss_value)

    return

if __name__ == "__main__":
    main()