import numpy as np
from scipy.ndimage.filters import median_filter

import code.core.metrics as metrics
from code.common import dict_to_struct


def post_process_output(input_struct):
    post_process_arousal_struct = post_process_emotion(input_struct.arousal)
    post_process_valence_struct = post_process_emotion(input_struct.valence)

    base_metric = (post_process_arousal_struct.base_metric + post_process_arousal_struct.base_metric) / 2.0
    best_metric = (post_process_arousal_struct.best_metric + post_process_arousal_struct.best_metric) / 2.0

    absolute_improvement = best_metric - base_metric

    if best_metric < base_metric:
        raise ValueError("This is not supposed to happen.")

    relative_improvement = get_relative_improvement(base_metric, best_metric)

    post_process_struct = dict()
    post_process_struct["arousal"] = post_process_arousal_struct
    post_process_struct["valence"] = post_process_valence_struct
    post_process_struct["absolute_improvement"] = absolute_improvement
    post_process_struct["relative_improvement"] = relative_improvement

    post_process_struct = dict_to_struct(post_process_struct)

    return post_process_struct


def post_process_emotion(input_struct):
    true = input_struct.true.copy(True)
    pred = input_struct.pred.copy(True)

    # Calculate reference stats.
    base_metric = metrics.batch_concordance_cc_numpy(np.reshape(true,
                                                                (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)),
                                                     np.reshape(pred,
                                                                (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)))
    best_filter_size = 1
    was_centred = False
    centre_offset = 0.0
    was_scaled = False
    scale_multiplier = 1.0
    best_time_shift = 0

    best_metric = base_metric

    # Median filtering.
    for filter_size in range(10, 510, 10):
        flat_pred = pred.reshape((input_struct.number_of_items,
                                  input_struct.batch_size * input_struct.seq_length))
        filtered_pred = median_filter(flat_pred, size=(1, filter_size))
        filtered_pred = filtered_pred.reshape((input_struct.batch_size * input_struct.number_of_items,
                                               input_struct.seq_length))

        current_metric = metrics.batch_concordance_cc_numpy(np.reshape(true, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)),
                                                      np.reshape(filtered_pred, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)))
        if current_metric > best_metric:
            best_filter_size = filter_size
            best_metric = current_metric

    if best_filter_size != 1:
        flat_pred = pred.reshape((input_struct.number_of_items,
                                  input_struct.batch_size * input_struct.seq_length))
        pred = median_filter(flat_pred, size=(1, best_filter_size))
        pred = pred.reshape((input_struct.batch_size * input_struct.number_of_items,
                             input_struct.seq_length))

    # Centering.
    flat_pred = pred.reshape((input_struct.number_of_items,
                              input_struct.batch_size * input_struct.seq_length))
    pred_mean = np.mean(flat_pred, axis=1).reshape((flat_pred.shape[0], 1))
    pred_mean = np.repeat(pred_mean, input_struct.batch_size, axis=1)
    pred_mean = pred_mean.reshape((input_struct.number_of_items * input_struct.batch_size, 1))

    true_mean = np.mean(true)

    centred_pred = flat_pred - pred_mean + true_mean

    current_metric = metrics.batch_concordance_cc_numpy(np.reshape(true, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)),
                                                  np.reshape(centred_pred, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)))
    if current_metric > best_metric:
        pred = centred_pred

        best_metric = current_metric

        was_centred = True
        centre_offset = true_mean
    else:
        centre_offset = 0.0

    # Scaling.
    flat_pred = pred.reshape((input_struct.number_of_items,
                              input_struct.batch_size * input_struct.seq_length))
    pred_std = np.std(flat_pred, axis=1).reshape((flat_pred.shape[0], 1))
    pred_std = np.repeat(pred_std, input_struct.batch_size, axis=1)
    pred_std = pred_std.reshape((input_struct.number_of_items * input_struct.batch_size, 1))

    true_std = np.std(true)

    scaled_pred = (flat_pred * true_std) / pred_std

    current_metric = metrics.batch_concordance_cc_numpy(np.reshape(true, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)),
                                                  np.reshape(scaled_pred, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)))

    if current_metric > best_metric:
        pred = scaled_pred

        best_metric = current_metric

        was_scaled = True
        scale_multiplier = true_std
    else:
        scale_multiplier = 1.0

    # Time-shifting.
    for time_shift in range(1, 255, 5):
        flat_pred = pred.reshape((input_struct.number_of_items,
                                  input_struct.batch_size * input_struct.seq_length))
        shifted_pred = shift(flat_pred, time_shift)
        shifted_pred = shifted_pred.reshape((input_struct.batch_size * input_struct.number_of_items,
                                             input_struct.seq_length))

        current_metric = metrics.batch_concordance_cc_numpy(np.reshape(true, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)),
                                                      np.reshape(shifted_pred, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)))
        if current_metric > best_metric:
            best_time_shift = time_shift
            best_metric = current_metric

    for time_shift in range(-1, -255, -5):
        flat_pred = pred.reshape((input_struct.number_of_items,
                                  input_struct.batch_size * input_struct.seq_length))
        shifted_pred = shift(flat_pred, time_shift)
        shifted_pred = shifted_pred.reshape((input_struct.batch_size * input_struct.number_of_items,
                                             input_struct.seq_length))

        current_metric = metrics.batch_concordance_cc_numpy(np.reshape(true, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)),
                                                      np.reshape(shifted_pred, (input_struct.number_of_items,
                                                                 input_struct.batch_size * input_struct.seq_length)))
        if current_metric > best_metric:
            best_time_shift = time_shift
            best_metric = current_metric

    # if best_time_shift != 0:
    #     flat_pred = pred.reshape((input_struct.number_of_items,
    #                               input_struct.batch_size * input_struct.seq_length))
    #     pred = shift(flat_pred, best_time_shift)
    #     pred = pred.reshape((input_struct.batch_size * input_struct.number_of_items,
    #                          input_struct.seq_length))

    absolute_improvement = best_metric - base_metric

    if best_metric < base_metric:
        raise ValueError("This is not supposed to happen.")

    relative_improvement = get_relative_improvement(base_metric, best_metric)

    output_struct = dict()
    output_struct["base_metric"] = base_metric
    output_struct["best_metric"] = best_metric
    output_struct["best_filter_size"] = best_filter_size
    output_struct["was_centred"] = was_centred
    output_struct["centre_offset"] = centre_offset
    output_struct["was_scaled"] = was_scaled
    output_struct["scale_multiplier"] = scale_multiplier
    output_struct["best_time_shift"] = best_time_shift
    output_struct["absolute_improvement"] = absolute_improvement
    output_struct["relative_improvement"] = relative_improvement

    output_struct = dict_to_struct(output_struct)

    return output_struct


def post_process_specific_emotion_specific_parameters(input_struct):
    pred = input_struct.pred.copy(True)

    # Median filtering.
    if input_struct.best_filter_size != 1:
        flat_pred = pred.reshape((input_struct.number_of_items,
                                  input_struct.batch_size * input_struct.seq_length))
        pred = median_filter(flat_pred, size=(1, input_struct.best_filter_size))
        pred = pred.reshape((input_struct.batch_size * input_struct.number_of_items,
                             input_struct.seq_length))

    # Centering.
    if input_struct.was_centred:
        flat_pred = pred.reshape((input_struct.number_of_items,
                                  input_struct.batch_size * input_struct.seq_length))
        pred_mean = np.mean(flat_pred, axis=1).reshape((flat_pred.shape[0], 1))
        pred_mean = np.repeat(pred_mean, input_struct.batch_size, axis=1)
        pred_mean = pred_mean.reshape((input_struct.number_of_items * input_struct.batch_size, 1))

        centred_pred = flat_pred - pred_mean + input_struct.centre_offset

        pred = centred_pred

    # Scaling.
    if input_struct.was_scaled:
        flat_pred = pred.reshape((input_struct.number_of_items,
                                  input_struct.batch_size * input_struct.seq_length))
        pred_std = np.std(flat_pred, axis=1).reshape((flat_pred.shape[0], 1))
        pred_std = np.repeat(pred_std, input_struct.batch_size, axis=1)
        pred_std = pred_std.reshape((input_struct.number_of_items * input_struct.batch_size, 1))

        scaled_pred = (flat_pred * input_struct.scale_multiplier) / pred_std

        pred = scaled_pred

    # Time-shifting.
    if input_struct.best_time_shift != 0:
        flat_pred = pred.reshape((input_struct.number_of_items,
                                  input_struct.batch_size * input_struct.seq_length))
        pred = shift(flat_pred, input_struct.best_time_shift)
        pred = pred.reshape((input_struct.batch_size * input_struct.number_of_items,
                             input_struct.seq_length))

    pred = pred.reshape((input_struct.batch_size * input_struct.number_of_items,
                         input_struct.seq_length, 1))
    return pred


def get_relative_improvement(base, new):
    rel_imp = ((new - base) / base) * 100.0
    return rel_imp


def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:, :n] = xs[:, 0].reshape((xs[:, 0].size, 1))
        e[:, n:] = xs[:, :-n]
    else:
        e[:, n:] = xs[:, -1].reshape((xs[:, -1].size, 1))
        e[:, :n] = xs[:, -n:]
    return e
