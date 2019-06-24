from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_end2end_model(modality):

    modality_to_model = {"audio": end2end_audio_model, }

    if modality in modality_to_model:
        model = modality_to_model[modality]
    else:
        raise ValueError('Requested name [{}] not a valid model'.format(modality))

    def wrapper(*args, **kwargs):
        return end2end_recurrent_model(model(*args), **kwargs)

    return wrapper


def end2end_audio_model(audio_frames,
                        is_training_dropout,
                        batch_size):
    _, seq_length, num_features = audio_frames.get_shape().as_list()
    audio_input = tf.reshape(audio_frames, [batch_size, num_features * seq_length, 1])

    net = tf.layers.conv1d(audio_input, 64, 8, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 10, 10)
    net = tf.layers.dropout(net, training=is_training_dropout)

    net = tf.layers.conv1d(net, 128, 6, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 8, 8)
    net = tf.layers.dropout(net, training=is_training_dropout)

    net = tf.layers.conv1d(net, 256, 6, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 8, 8)
    net = tf.layers.dropout(net, training=is_training_dropout)

    net = tf.reshape(net, [batch_size, seq_length, 256])  # 256])
    return net


def end2end_recurrent_model(net,
                            num_layers,
                            hidden_units,
                            dropout_keep_prob,
                            number_of_outputs,
                            batch_size):
    _, seq_length, num_features = net.get_shape().as_list()

    def _get_cell(l_no):
        lstm = tf.contrib.rnn.LSTMCell(hidden_units,
                                       use_peepholes=True,
                                       cell_clip=100,
                                       state_is_tuple=True)
        if dropout_keep_prob < 1.0:
            if l_no == 0:
                lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     input_keep_prob=dropout_keep_prob,
                                                     output_keep_prob=dropout_keep_prob,
                                                     state_keep_prob=dropout_keep_prob,
                                                     variational_recurrent=True,
                                                     input_size=num_features,
                                                     dtype=tf.float32)
            else:
                lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     output_keep_prob=dropout_keep_prob,
                                                     state_keep_prob=dropout_keep_prob,
                                                     variational_recurrent=True,
                                                     input_size=hidden_units,
                                                     dtype=tf.float32)
        return lstm

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no) for l_no in range(num_layers)], state_is_tuple=True)

    outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

    if seq_length is None:
        seq_length = -1

    net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

    mean_prediction = tf.layers.dense(net, number_of_outputs)
    mean_prediction = tf.reshape(mean_prediction, (batch_size, seq_length, number_of_outputs))

    var_prediction = tf.layers.dense(net, number_of_outputs)
    var_prediction = tf.reshape(var_prediction, (batch_size, seq_length, number_of_outputs))

    return mean_prediction, var_prediction
