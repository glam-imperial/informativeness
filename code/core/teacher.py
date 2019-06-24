import tensorflow as tf


def teacher(source_pred_expected,
            source_pred_aleatory,
            source_pred_epistemic,
            true_mean,
            true_var,
            number_of_outputs,
            is_training_dropout,
            batch_size,
            use_aleatory,
            use_uncertainty):
    _, seq_length, num_features = source_pred_expected.get_shape().as_list()

    if not use_uncertainty:
        net = tf.stack([true_mean,
                        true_var,
                        true_mean - source_pred_expected],
                       axis=2)
        net = tf.reshape(net, shape=(batch_size * seq_length, 1, 2 * 3, 1))
        net = tf.layers.conv2d(net, 24, (1, 2 * 3), padding='valid')
        net = tf.layers.dropout(net, training=is_training_dropout)
        net = tf.reshape(net, shape=(batch_size * seq_length, 24))
        net = tf.layers.dense(net, number_of_outputs)
        net = tf.reshape(net, shape=(batch_size, seq_length, number_of_outputs))
    else:
        if use_aleatory:
            net = tf.stack([source_pred_expected,
                            source_pred_aleatory,
                            source_pred_epistemic,
                            true_mean,
                            true_var,
                            true_mean - source_pred_expected],
                           axis=2)
            net = tf.reshape(net, shape=(batch_size * seq_length, 1, 2 * 6, 1))
            net = tf.layers.conv2d(net, 24, (1, 2 * 6), padding='valid')
            net = tf.layers.dropout(net, training=is_training_dropout)
            net = tf.reshape(net, shape=(batch_size * seq_length, 24))
            net = tf.layers.dense(net, number_of_outputs)
            net = tf.reshape(net, shape=(batch_size, seq_length, number_of_outputs))
        else:
            net = tf.stack([source_pred_expected,
                            source_pred_epistemic,
                            true_mean,
                            true_var,
                            true_mean - source_pred_expected],
                           axis=2)
            net = tf.reshape(net, shape=(batch_size * seq_length, 1, 2 * 5, 1))
            net = tf.layers.conv2d(net, 24, (1, 2 * 5), padding='valid')
            net = tf.layers.dropout(net, training=is_training_dropout)
            net = tf.reshape(net, shape=(batch_size * seq_length, 24))
            net = tf.layers.dense(net, number_of_outputs)
            net = tf.reshape(net, shape=(batch_size, seq_length, number_of_outputs))

    weights = net

    return weights
