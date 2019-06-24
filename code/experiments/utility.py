import numpy as np
import tensorflow as tf

from code.common import dict_to_struct


def flatten_emotion_data(data, emotion, flattened_size):
    flattened_data = tf.reshape(data[:, :, emotion],
                                (-1,))
    flattened_data = tf.reshape(flattened_data,
                                (flattened_size, 1, 1, 1))
    return flattened_data


def replace_dict_value(input_dict, old_value, new_value):
    for k, v in input_dict.items():
        if isinstance(v, str):
            if v == old_value:
                input_dict[k] = np.nan_to_num(new_value, copy=True)
    return input_dict


def contains_nan(array):
    return np.isnan(array).any()


def run_epoch(config_struct):
    sess = config_struct.sess
    init_op = config_struct.init_op
    steps_per_epoch = config_struct.steps_per_epoch
    next_element = config_struct.next_element
    batch_size = config_struct.batch_size
    seq_length = config_struct.seq_length
    input_gaussian_noise = config_struct.input_gaussian_noise
    has_meta = config_struct.has_meta
    mc_samples = config_struct.mc_samples
    get_vars = config_struct.get_vars
    input_feed_dict = config_struct.feed_dict
    saver = config_struct.saver

    out_tf = list()
    track_var = list()
    var_names = list()
    counter = 0
    for t in get_vars:
        out_tf.append(t[0])
        track_var.append(t[1])
        var_names.append(t[2])
        counter += 1

    # Initialize an iterator over the dataset split.
    sess.run(init_op)

    # Store variable sequence.
    stored_variables = dict()
    for emotion in ["arousal", "valence"]:
        stored_variables[emotion] = dict()
        stored_variables[emotion]["true"] = np.empty((steps_per_epoch * batch_size,
                                                      seq_length),
                                                     dtype=np.float32)
        stored_variables[emotion]["true_var"] = np.empty((steps_per_epoch * batch_size,
                                                          seq_length),
                                                         dtype=np.float32)

    for i, track in enumerate(track_var):
        if track in ["yes", "yes_mc"]:
            for emotion in ["arousal", "valence"]:
                stored_variables[emotion][var_names[i]] = np.empty((steps_per_epoch * batch_size,
                                                                    seq_length),
                                                                   dtype=np.float32)

        if track == "yes_mc":
            for emotion in ["arousal", "valence"]:
                stored_variables[emotion][var_names[i] + "_epi"] = np.empty((steps_per_epoch * batch_size,
                                                                             seq_length),
                                                                            dtype=np.float32)
        if track == "loss":
            stored_variables[var_names[i]] = list()

    if has_meta:
        source_subsequences = dict()
        for k, v in config_struct.source_vars.items():
            source_subsequences[k] = np.empty((batch_size,
                                               seq_length,
                                               2),
                                              dtype=np.float32)

        source_subject_to_id = config_struct.source_subject_to_id

    temp_mc = dict()
    temp = dict()
    for i, track in enumerate(track_var):
        if track in ["yes", "yes_mc"]:
            temp_mc[var_names[i]] = np.empty((batch_size,
                                                  seq_length,
                                                  2,
                                                  mc_samples),
                                                 dtype=np.float32)
            temp[var_names[i]] = None
        if track == "yes_mc":
            temp[var_names[i] + "_epi"] = None

    subject_to_id = dict()
    for step in range(steps_per_epoch):
        batch_tuple = sess.run(next_element)
        sample_id = batch_tuple["sample_id"]
        subject_id = batch_tuple["subject_id"]
        try:
            label = batch_tuple["label"]
            label_shape = batch_tuple["label_shape"]
            original_labels = True
        except KeyError:
            label = batch_tuple["gs_label"]
            label_shape = batch_tuple["gs_label_shape"]
            original_labels = False
        raw_audio = batch_tuple["raw_audio"]
        raw_audio_shape = batch_tuple["raw_audio_shape"]
        audio_features_numpy = batch_tuple["audio_features"][:, :, :178]
        image_features_appearance_numpy = batch_tuple["image_features_appearance"]
        image_features_geometric_numpy = batch_tuple["image_features_geometric"]

        subject_to_id[subject_id[0, 0][0]] = step

        seq_pos_start = step * batch_size
        seq_pos_end = seq_pos_start + batch_size

        # Augment data.
        jitter = np.random.normal(scale=input_gaussian_noise,
                                  size=raw_audio.shape)
        raw_audio_plus_jitter = raw_audio + jitter
        jitter = np.random.normal(scale=input_gaussian_noise,
                                  size=audio_features_numpy.shape)
        audio_features_numpy_plus_jitter = audio_features_numpy + jitter
        jitter = np.random.normal(scale=input_gaussian_noise,
                                  size=image_features_appearance_numpy.shape)
        image_features_appearance_numpy_plus_jitter = image_features_appearance_numpy + jitter
        jitter = np.random.normal(scale=input_gaussian_noise,
                                  size=image_features_geometric_numpy.shape)
        image_features_geometric_numpy_plus_jitter = image_features_geometric_numpy + jitter

        if has_meta:
            for sample in range(batch_size):
                source_subject_id = source_subject_to_id[subject_id[sample, 0][0]]
                sample_start_id = sample_id[sample, 0][0]
                sample_end_id = sample_id[sample, -1][0]

                for k, v in config_struct.source_vars.items():
                    source_subsequences[k][sample, :, :] = v[source_subject_id, sample_start_id:sample_end_id + 1, :]

        feed_dict = {k: v for k, v in input_feed_dict.items()}
        if original_labels:
            feed_dict = replace_dict_value(feed_dict, "true_raw", label)
        feed_dict = replace_dict_value(feed_dict, "audio", raw_audio_plus_jitter)
        feed_dict = replace_dict_value(feed_dict, "audio_features", audio_features_numpy_plus_jitter)
        feed_dict = replace_dict_value(feed_dict, "image_features_appearance", image_features_appearance_numpy_plus_jitter)
        feed_dict = replace_dict_value(feed_dict, "image_features_geometric", image_features_geometric_numpy_plus_jitter)
        if has_meta:
            feed_dict = replace_dict_value(feed_dict, "source_pred_mean_input", source_subsequences["pred"])
            feed_dict = replace_dict_value(feed_dict, "source_pred_var_input", source_subsequences["pred_ale"])
            feed_dict = replace_dict_value(feed_dict, "source_pred_mean_epistemic_input", source_subsequences["pred_epi"])

        for t in range(mc_samples):
            out_np = sess.run(out_tf,
                              feed_dict=feed_dict)
            for i, track in enumerate(track_var):
                if track in ["yes", "yes_mc"]:
                    temp_mc[var_names[i]][:, :, :, t] = out_np[i]

        for i, track in enumerate(track_var):
            if track in ["yes", "yes_mc"]:
                temp[var_names[i]] = np.mean(temp_mc[var_names[i]], axis=3)
            if track == "yes_mc":
                if mc_samples > 1:
                    temp[var_names[i] + "_epi"] = np.var(temp_mc[var_names[i]], axis=3)
                else:
                    temp[var_names[i] + "_epi"] = temp_mc[var_names[i]].reshape((batch_size,
                                                  seq_length,
                                                  2))
        if original_labels:
            stored_variables["arousal"]["true"][seq_pos_start:seq_pos_end, :] = np.mean(label[:, :, :6], axis=2)
            stored_variables["arousal"]["true_var"][seq_pos_start:seq_pos_end, :] = np.var(label[:, :, :6], axis=2)
            stored_variables["valence"]["true"][seq_pos_start:seq_pos_end, :] = np.mean(label[:, :, 6:], axis=2)
            stored_variables["valence"]["true_var"][seq_pos_start:seq_pos_end, :] = np.var(label[:, :, 6:], axis=2)
        else:
            stored_variables["arousal"]["true"][seq_pos_start:seq_pos_end, :] = label[:, :, 0]
            stored_variables["valence"]["true"][seq_pos_start:seq_pos_end, :] = label[:, :, 1]

        for i, track in enumerate(track_var):
            if track in ["yes", "yes_mc"]:
                for e, emotion in enumerate(["arousal", "valence"]):
                    stored_variables[emotion][var_names[i]][seq_pos_start:seq_pos_end, :] = temp[var_names[i]][:, :, e]
            if track == "yes_mc":
                for e, emotion in enumerate(["arousal", "valence"]):
                    stored_variables[emotion][var_names[i] + "_epi"][seq_pos_start:seq_pos_end, :] = temp[var_names[i] + "_epi"][:, :, e]
            if track == "loss":
                stored_variables[var_names[i]].append(out_np[i])

    for i, track in enumerate(track_var):
        if track == "loss":
            stored_variables[var_names[i]] = np.mean(np.array(stored_variables[var_names[i]]))

    if saver is not None:
        for path, s in saver.items():
            s.save(sess, path)

    for emotion in ["arousal", "valence"]:
        stored_variables[emotion] = dict_to_struct(stored_variables[emotion])
    stored_variables = dict_to_struct(stored_variables)

    return stored_variables, subject_to_id
