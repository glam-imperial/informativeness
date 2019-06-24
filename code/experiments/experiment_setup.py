from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect

import numpy as np
import tensorflow as tf

import code.core.models as models
import code.core.metrics as metrics
import code.core.teacher as policy
import code.core.post_process as post_process
import code.dataset_read.data_provider as data_provider
import code.core.losses as losses
from code.experiments.utility import flatten_emotion_data, run_epoch
from code.common import dict_to_struct, make_dirs_safe


def train(configuration):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = repr(configuration.GPU)

    ####################################################################################################################
    # Interpret configuration arguments.
    ####################################################################################################################
    if configuration.uncertainty in ["epistemic", "both"]:
        is_epistemic = True
        number_of_MC_samples = configuration.number_of_mc_samples
        dropout_keep_prob_eff = configuration.dropout_keep_prob
    else:
        is_epistemic = False
        number_of_MC_samples = 1
        dropout_keep_prob_eff = 1.0

    if configuration.meta_uncertainty in ["epistemic", "epistemic_only", "both"]:
        number_of_meta_MC_samples = configuration.number_of_meta_MC_samples
    else:
        number_of_meta_MC_samples = 1

    train_steps_per_epoch = (configuration.total_seq_length * configuration.train_size) //\
                            (configuration.seq_length * configuration.train_batch_size)
    val_steps_per_epoch = (configuration.total_seq_length * configuration.valid_size) //\
                          (configuration.total_seq_length * configuration.valid_batch_size)

    if configuration.has_meta:
        file_path_suffix = "meta_" + \
                           configuration.meta_uncertainty + "_" +\
                           "model_" +\
                           configuration.framework + "_" + \
                           configuration.modality + "_" + \
                           configuration.loss_measure_name + "_" + \
                           configuration.variance_labels + "_" + \
                           repr(configuration.trial)
    else:
        file_path_suffix = configuration.framework + "_" + \
                           configuration.modality + "_" + \
                           configuration.uncertainty + "_" + \
                           configuration.loss_measure_name + "_" + \
                           configuration.variance_labels + "_" + \
                           repr(configuration.trial)

    data_folder = configuration.target_data_folder

    train_dir = data_folder + "/ckpt/train/" + file_path_suffix
    checkpoint_dir = train_dir
    log_dir = data_folder + "/ckpt/log/" + file_path_suffix
    test_pred_dir = log_dir + "/test_pred"
    results_log_file = data_folder + "/losses/" + file_path_suffix

    base_model_path_last = log_dir + "/base_last"
    base_model_path_best = log_dir + "/base"
    source_model_path_last = log_dir + "/source_last"
    source_model_path_best = log_dir + "/source"
    teacher_model_path_last = log_dir + "/teacher_last"
    teacher_model_path_best = log_dir + "/teacher"

    starting_epoch = 0
    starting_best_performance = - 1.0

    ####################################################################################################################
    # Form computational graph.
    ####################################################################################################################
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            ############################################################################################################
            # Get dataset iterators.
            ############################################################################################################
            dataset_train = data_provider.get_split(configuration.data_folder,
                                                    is_training=True,
                                                    split_name="train",
                                                    batch_size=configuration.train_batch_size,
                                                    seq_length=configuration.seq_length,
                                                    buffer_size=train_steps_per_epoch + 1)
            dataset_train_full = data_provider.get_split(configuration.data_folder,
                                                         is_training=False,
                                                         split_name="train",
                                                         batch_size=configuration.valid_batch_size,
                                                         seq_length=configuration.total_seq_length,
                                                         buffer_size=10)
            dataset_valid = data_provider.get_split(configuration.data_folder,
                                                    is_training=False,
                                                    split_name="valid",
                                                    batch_size=configuration.valid_batch_size,
                                                    seq_length=configuration.total_seq_length,
                                                    buffer_size=10)
            dataset_test = data_provider.get_split(configuration.data_folder,
                                                   is_training=False,
                                                   split_name="test",
                                                   batch_size=configuration.valid_batch_size,
                                                   seq_length=configuration.total_seq_length,
                                                   buffer_size=10)

            iterator_train = tf.data.Iterator.from_structure(dataset_train.output_types,
                                                             dataset_train.output_shapes)
            iterator_train_full = tf.data.Iterator.from_structure(dataset_train_full.output_types,
                                                                  dataset_train_full.output_shapes)
            iterator_valid = tf.data.Iterator.from_structure(dataset_valid.output_types,
                                                             dataset_valid.output_shapes)
            iterator_test = tf.data.Iterator.from_structure(dataset_test.output_types,
                                                            dataset_test.output_shapes)

            next_element_train = iterator_train.get_next()
            next_element_train_full = iterator_train_full.get_next()
            next_element_valid = iterator_valid.get_next()
            next_element_test = iterator_test.get_next()

            init_op_train = iterator_train.make_initializer(dataset_train)
            init_op_train_full = iterator_train_full.make_initializer(dataset_train_full)
            init_op_valid = iterator_valid.make_initializer(dataset_valid)
            init_op_test = iterator_test.make_initializer(dataset_test)

            ############################################################################################################
            # Define placeholders.
            ############################################################################################################
            batch_size_tensor = tf.placeholder(tf.int32)
            # seq_length_tensor = tf.placeholder(tf.int32)

            sample_ids_train = tf.placeholder(tf.int32, (None, configuration.seq_length))
            subject_ids_train = tf.placeholder(tf.int32, (None, configuration.seq_length))
            true_raw_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 12))
            audio_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 640))
            # audio_features_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 534))
            audio_features_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 178))
            image_features_appearance_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 1014))
            image_features_geometric_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 3798))

            sample_ids_test = tf.placeholder(tf.int32, (None, configuration.total_seq_length))
            subject_ids_test = tf.placeholder(tf.int32, (None, configuration.total_seq_length))
            true_raw_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 12))
            audio_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 640))
            # audio_features_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 534))
            audio_features_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 178))
            image_features_appearance_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 1014))
            image_features_geometric_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 3798))

            ############################################################################################################
            # Meta-learning placeholders.
            ############################################################################################################
            source_pred_mean_input_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 2))
            source_pred_var_input_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 2))
            source_pred_mean_epistemic_input_train = tf.placeholder(tf.float32, (None, configuration.seq_length, 2))

            source_pred_mean_input_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 2))
            source_pred_var_input_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 2))
            source_pred_mean_epistemic_input_test = tf.placeholder(tf.float32, (None, configuration.total_seq_length, 2))

            ############################################################################################################
            # Other placeholders.
            ############################################################################################################
            is_training_dropout_tensor = tf.placeholder(tf.bool, shape=[])
            is_training_batchnorm_tensor = tf.placeholder(tf.bool, shape=[])

            true_arousal_mean_train, true_arousal_var_train = tf.nn.moments(true_raw_train[:, :, 0:6], axes=[2])
            true_valence_mean_train, true_valence_var_train = tf.nn.moments(true_raw_train[:, :, 6:12], axes=[2])

            true_mean_train = tf.stack([true_arousal_mean_train, true_valence_mean_train], axis=2)
            true_var_train = tf.stack([true_arousal_var_train, true_valence_var_train], axis=2)

            true_arousal_mean_test, true_arousal_var_test = tf.nn.moments(true_raw_test[:, :, 0:6], axes=[2])
            true_valence_mean_test, true_valence_var_test = tf.nn.moments(true_raw_test[:, :, 6:12], axes=[2])

            true_mean_test = tf.stack([true_arousal_mean_test, true_valence_mean_test], axis=2)
            true_var_test = tf.stack([true_arousal_var_test, true_valence_var_test], axis=2)

            ############################################################################################################
            # Define model graph and get model.
            ############################################################################################################
            # Select model framework.
            get_model_framework = models.get_end2end_model

            if configuration.has_meta:
                with tf.variable_scope("Source"):
                    source_pred_mean_train, \
                    source_pred_var_train = get_model_framework(configuration.modality)(audio_train,
                                                                                        is_training_dropout_tensor,
                                                                                        batch_size_tensor,
                                                                                        num_layers=configuration.num_layers,
                                                                                        hidden_units=configuration.hidden_units,
                                                                                        dropout_keep_prob=dropout_keep_prob_eff,
                                                                                        number_of_outputs=2,
                                                                                        batch_size=batch_size_tensor)
                with tf.variable_scope("Source", reuse=True):
                    source_pred_mean_test, \
                    source_pred_var_test = get_model_framework(configuration.modality)(audio_test,
                                                                                       is_training_dropout_tensor,
                                                                                       batch_size_tensor,
                                                                                       num_layers=configuration.num_layers,
                                                                                       hidden_units=configuration.hidden_units,
                                                                                       dropout_keep_prob=dropout_keep_prob_eff,
                                                                                       number_of_outputs=2,
                                                                                       batch_size=batch_size_tensor)

                with tf.variable_scope("Teacher"):
                    if configuration.meta_uncertainty in ["aleatory", "both"]:
                        use_aleatory = True
                    else:
                        use_aleatory = False

                    if configuration.meta_uncertainty == "none":
                        use_uncertainty = False
                    else:
                        use_uncertainty = True
                    teacher_weights_train = policy.teacher(source_pred_mean_input_train,
                                                           source_pred_var_input_train,
                                                           source_pred_mean_epistemic_input_train,
                                                           true_mean_train,
                                                           true_var_train,
                                                           number_of_outputs=2,
                                                           is_training_dropout=is_training_dropout_tensor,
                                                           batch_size=batch_size_tensor,
                                                           use_aleatory=use_aleatory,
                                                           use_uncertainty=use_uncertainty)
                with tf.variable_scope("Teacher", reuse=True):
                    if configuration.meta_uncertainty in ["aleatory", "both"]:
                        use_aleatory = True
                    else:
                        use_aleatory = False

                    if configuration.meta_uncertainty == "none":
                        use_uncertainty = False
                    else:
                        use_uncertainty = True
                    teacher_weights_test = policy.teacher(source_pred_mean_input_test,
                                                          source_pred_var_input_test,
                                                          source_pred_mean_epistemic_input_test,
                                                          true_mean_test,
                                                          true_var_test,
                                                          number_of_outputs=2,
                                                          is_training_dropout=is_training_dropout_tensor,
                                                          batch_size=batch_size_tensor,
                                                          use_aleatory=use_aleatory,
                                                          use_uncertainty=use_uncertainty)
            else:
                source_pred_mean_train = None
                source_pred_var_train = None
                teacher_weights_train = None

                source_pred_mean_test = None
                source_pred_var_test = None
                teacher_weights_test = None

            with tf.variable_scope("Base"):
                base_pred_mean_train, \
                base_pred_var_train = get_model_framework(configuration.modality)(audio_train,
                                                                                  is_training_dropout_tensor,
                                                                                  batch_size_tensor,
                                                                                  num_layers=configuration.num_layers,
                                                                                  hidden_units=configuration.hidden_units,
                                                                                  dropout_keep_prob=dropout_keep_prob_eff,
                                                                                  number_of_outputs=2,
                                                                                  batch_size=batch_size_tensor)

            with tf.variable_scope("Base", reuse=True):
                base_pred_mean_test, \
                base_pred_var_test = get_model_framework(configuration.modality)(audio_test,
                                                                                 is_training_dropout_tensor,
                                                                                 batch_size_tensor,
                                                                                 num_layers=configuration.num_layers,
                                                                                 hidden_units=configuration.hidden_units,
                                                                                 dropout_keep_prob=dropout_keep_prob_eff,
                                                                                 number_of_outputs=2,
                                                                                 batch_size=batch_size_tensor)

            base_emotion_losses = list()
            source_emotion_losses = list()
            teacher_emotion_losses = list()
            loss_function = losses.get_loss_function(configuration.uncertainty)
            loss_function_argnames = inspect.getargspec(loss_function)[0]
            for i, name in enumerate(['arousal', 'valence']):
                mc_tensor_shape_train = [batch_size_tensor, configuration.seq_length]
                flattened_size_train = mc_tensor_shape_train[0] * mc_tensor_shape_train[1]

                base_single_pred_mean_train = flatten_emotion_data(base_pred_mean_train,
                                                                   i,
                                                                   flattened_size_train)

                base_single_pred_var_train = flatten_emotion_data(base_pred_var_train,
                                                                  i,
                                                                  flattened_size_train)

                if configuration.has_meta:
                    source_single_pred_mean_train = flatten_emotion_data(source_pred_mean_train,
                                                                         i,
                                                                         flattened_size_train)

                    source_single_pred_var_input_train = flatten_emotion_data(source_pred_var_input_train,
                                                                              i,
                                                                              flattened_size_train)
                    source_single_pred_var_train = flatten_emotion_data(source_pred_var_train,
                                                                        i,
                                                                        flattened_size_train)

                    teacher_single_weights_train = flatten_emotion_data(teacher_weights_train,
                                                                        i,
                                                                        flattened_size_train)

                else:
                    source_single_pred_mean_train = None
                    source_single_pred_var_input_train = None
                    source_single_pred_var_train = None

                single_true_mean_train = flatten_emotion_data(true_mean_train,
                                                              i,
                                                              flattened_size_train)
                single_true_var_train = flatten_emotion_data(true_var_train,
                                                             i,
                                                             flattened_size_train)

                if configuration.has_meta:
                    base_loss = losses.aleatory_attenuated_concordance_cc(base_single_pred_mean_train,
                                                                          teacher_single_weights_train,
                                                                          single_true_mean_train)
                    base_loss_flat = losses.aleatory_attenuated_concordance_cc(base_single_pred_mean_train,
                                                                               tf.ones_like(teacher_single_weights_train),
                                                                               single_true_mean_train)
                    base_loss_ale = losses.aleatory_attenuated_concordance_cc(base_single_pred_mean_train,
                                                                              source_single_pred_var_input_train,
                                                                              single_true_mean_train)

                    base_loss_flat = tf.cond(base_loss_flat < base_loss_ale,
                                             lambda: base_loss_flat,
                                             lambda: base_loss_ale)

                else:
                    loss_kwargs = dict()
                    loss_kwargs["pred"] = base_single_pred_mean_train
                    loss_kwargs["log_var_pred"] = base_single_pred_var_train
                    loss_kwargs["true"] = single_true_mean_train
                    loss_kwargs["var_true"] = single_true_var_train
                    loss_kwargs["batch_size"] = configuration.train_batch_size
                    loss_kwargs["seq_length"] = configuration.seq_length
                    loss_kwargs = {kw: loss_kwargs[kw] for kw in loss_function_argnames}

                    base_loss = loss_function(**loss_kwargs)

                    base_loss_flat = 0.0

                base_emotion_losses.append(base_loss / 2.0)
                teacher_emotion_losses.append((base_loss - base_loss_flat) / 2.0)

                if configuration.has_meta:
                    if configuration.meta_uncertainty == "epistemic_only":
                        source_loss = losses.concordance_cc(source_single_pred_mean_train,
                                                            single_true_mean_train)
                    else:
                        source_loss = losses.aleatory_attenuated_concordance_cc(source_single_pred_mean_train,
                                                                                source_single_pred_var_train,
                                                                                single_true_mean_train)
                    source_emotion_losses.append(source_loss / 2.0)

            vars = tf.trainable_variables()
            base_vars = [v for v in vars if v.name.startswith("Base")]
            source_vars = [v for v in vars if v.name.startswith("Source")]
            teacher_vars = [v for v in vars if v.name.startswith("Teacher")]

            base_saver = tf.train.Saver({v.name: v for v in vars if v.name.startswith("Base")})
            if configuration.has_meta:
                source_saver = tf.train.Saver({v.name: v for v in vars if v.name.startswith("Source")})
                teacher_saver = tf.train.Saver({v.name: v for v in vars if v.name.startswith("Teacher")})

            base_total_loss = tf.reduce_sum(tf.stack(base_emotion_losses[-2:]))
            base_optimizer = tf.train.AdamOptimizer(configuration.initial_learning_rate).minimize(base_total_loss,
                                                                                                  var_list=base_vars)
            if configuration.has_meta:
                source_total_loss = tf.reduce_sum(tf.stack(source_emotion_losses[-2:]))
                source_optimizer = tf.train.AdamOptimizer(configuration.initial_learning_rate).minimize(source_total_loss,
                                                                                                        var_list=source_vars)

                teacher_total_loss = tf.reduce_sum(tf.stack(teacher_emotion_losses[-2:]))
                teacher_optimizer = tf.train.AdamOptimizer(configuration.initial_learning_rate).minimize(
                    teacher_total_loss,
                    var_list=teacher_vars)
            else:
                source_total_loss = None
                source_optimizer = None
                teacher_total_loss = None
                teacher_optimizer = None

            ############################################################################################################
            # Initialize variables and perform experiment.
            ############################################################################################################
            sess.run(tf.global_variables_initializer())

            ############################################################################################################
            # Train source model.
            ############################################################################################################
            print("Section: Source model.")
            if configuration.has_meta:
                print("Training source model.")
                for epoch in range(configuration.meta_num_epochs):
                    # Train source model.
                    config_epoch_pass = dict()
                    config_epoch_pass["sess"] = sess
                    config_epoch_pass["init_op"] = init_op_train
                    config_epoch_pass["steps_per_epoch"] = train_steps_per_epoch
                    config_epoch_pass["next_element"] = next_element_train
                    config_epoch_pass["batch_size"] = configuration.train_batch_size
                    config_epoch_pass["seq_length"] = configuration.seq_length
                    config_epoch_pass["input_gaussian_noise"] = configuration.input_gaussian_noise
                    config_epoch_pass["has_meta"] = False
                    config_epoch_pass["mc_samples"] = 1
                    config_epoch_pass["get_vars"] = [(source_optimizer, "no", None),
                                                             (source_total_loss, "loss", "loss"),
                                                             (source_pred_mean_train, "yes", "pred")]
                    config_epoch_pass["feed_dict"] = {batch_size_tensor: configuration.train_batch_size,
                                                              is_training_dropout_tensor: True,
                                                              is_training_batchnorm_tensor: True,
                                                              true_raw_train: "true_raw",
                                                              audio_train: "audio",
                                                              audio_features_train: "audio_features",
                                                              image_features_appearance_train: "image_features_appearance",
                                                              image_features_geometric_train: "image_features_geometric"}
                    config_epoch_pass["saver"] = {source_model_path_last: source_saver}

                    config_epoch_pass = dict_to_struct(config_epoch_pass)

                    source_train_items, source_train_subject_to_id = run_epoch(config_epoch_pass)

                    source_saver.save(sess, source_model_path_best)

                print("Getting source model predictions.")
                source_saver.restore(sess, source_model_path_best)
                config_epoch_pass = dict()
                config_epoch_pass["sess"] = sess
                config_epoch_pass["init_op"] = init_op_train_full
                config_epoch_pass["steps_per_epoch"] = val_steps_per_epoch
                config_epoch_pass["next_element"] = next_element_train_full
                config_epoch_pass["batch_size"] = configuration.valid_batch_size
                config_epoch_pass["seq_length"] = configuration.total_seq_length
                config_epoch_pass["input_gaussian_noise"] = configuration.input_gaussian_noise
                config_epoch_pass["has_meta"] = False
                config_epoch_pass["mc_samples"] = number_of_meta_MC_samples
                config_epoch_pass["get_vars"] = [(source_pred_mean_test, "yes_mc", "pred"),
                                                 (source_pred_var_test, "yes", "pred_ale")]
                config_epoch_pass["feed_dict"] = {batch_size_tensor: configuration.valid_batch_size,
                                                  is_training_dropout_tensor: True,
                                                  is_training_batchnorm_tensor: False,
                                                  true_raw_test: "true_raw",
                                                  audio_test: "audio",
                                                  audio_features_test: "audio_features",
                                                  image_features_appearance_test: "image_features_appearance",
                                                  image_features_geometric_test: "image_features_geometric"}
                config_epoch_pass["saver"] = None

                config_epoch_pass = dict_to_struct(config_epoch_pass)

                src_items, src_subject_to_id = run_epoch(config_epoch_pass)

                src_items_pp = src_items

                src_item_pred = np.stack([src_items_pp.arousal.pred, src_items_pp.valence.pred], axis=2)
                src_item_pred_ale = np.stack([src_items_pp.arousal.pred_ale, src_items_pp.valence.pred_ale],
                                             axis=2)
                src_item_pred_epi = np.stack([src_items_pp.arousal.pred_epi, src_items_pp.valence.pred_epi],
                                             axis=2)

            ############################################################################################################
            # Train base model.
            ############################################################################################################
            print("Section: Base model.")
            make_dirs_safe(test_pred_dir)
            print("Start training base model.")
            print("Fresh base model.")
            losses_fp = open(results_log_file, "w")
            losses_fp.close()

            for ee, epoch in enumerate(range(starting_epoch, configuration.num_epochs + starting_epoch)):
                print("Train Base model.")

                config_epoch_pass = dict()
                config_epoch_pass["sess"] = sess
                config_epoch_pass["init_op"] = init_op_train
                config_epoch_pass["steps_per_epoch"] = train_steps_per_epoch
                config_epoch_pass["next_element"] = next_element_train
                config_epoch_pass["batch_size"] = configuration.train_batch_size
                config_epoch_pass["seq_length"] = configuration.seq_length
                config_epoch_pass["input_gaussian_noise"] = configuration.input_gaussian_noise
                config_epoch_pass["has_meta"] = configuration.has_meta
                config_epoch_pass["mc_samples"] = 1
                config_epoch_pass["get_vars"] = [(base_optimizer, "no", None),
                                                     (base_total_loss, "loss", "loss"),
                                                     (base_pred_mean_train, "yes", "pred")]
                config_epoch_pass["feed_dict"] = {batch_size_tensor: configuration.train_batch_size,
                                                      is_training_dropout_tensor: True,
                                                      is_training_batchnorm_tensor: True,
                                                      true_raw_train: "true_raw",
                                                      audio_train: "audio",
                                                      audio_features_train: "audio_features",
                                                      image_features_appearance_train: "image_features_appearance",
                                                      image_features_geometric_train: "image_features_geometric"}
                if configuration.has_meta:
                    config_epoch_pass["get_vars"].append((teacher_optimizer, "no", "None"))
                    config_epoch_pass["get_vars"].append((teacher_weights_train, "yes", "informativeness"))

                    config_epoch_pass["feed_dict"][source_pred_mean_input_train] = "source_pred_mean_input"
                    config_epoch_pass["feed_dict"][source_pred_var_input_train] = "source_pred_var_input"
                    config_epoch_pass["feed_dict"][source_pred_mean_epistemic_input_train] = "source_pred_mean_epistemic_input"

                    config_epoch_pass["source_vars"] = {"pred": src_item_pred,
                                                            "pred_ale": src_item_pred_ale,
                                                            "pred_epi": src_item_pred_epi}
                    config_epoch_pass["source_subject_to_id"] = src_subject_to_id

                config_epoch_pass["saver"] = {base_model_path_last: base_saver}
                if configuration.has_meta:
                    config_epoch_pass["saver"][teacher_model_path_last] = teacher_saver

                config_epoch_pass = dict_to_struct(config_epoch_pass)

                base_train_items, base_train_subject_to_id = run_epoch(config_epoch_pass)

                config_post_processing = dict()
                for emotion in ["arousal", "valence"]:
                    config_post_processing[emotion] = dict()
                    config_post_processing[emotion]["number_of_items"] = configuration.valid_size
                    config_post_processing[emotion]["batch_size"] = configuration.valid_batch_size
                    config_post_processing[emotion]["seq_length"] = configuration.total_seq_length
                config_post_processing["arousal"]["true"] = base_train_items.arousal.true
                config_post_processing["arousal"]["pred"] = base_train_items.arousal.pred
                config_post_processing["valence"]["true"] = base_train_items.valence.true
                config_post_processing["valence"]["pred"] = base_train_items.valence.pred
                for emotion in ["arousal", "valence"]:
                    config_post_processing[emotion] = dict_to_struct(config_post_processing[emotion])

                config_post_processing = dict_to_struct(config_post_processing)

                base_train_postprocess_struct = post_process.post_process_output(config_post_processing)

                if ee == 0:
                    best_performance = starting_best_performance

                if (ee+1) % configuration.val_every_n_epoch == 0:
                    print("Valid Base model.")
                    config_epoch_pass = dict()
                    config_epoch_pass["sess"] = sess
                    config_epoch_pass["init_op"] = init_op_valid
                    config_epoch_pass["steps_per_epoch"] = val_steps_per_epoch
                    config_epoch_pass["next_element"] = next_element_valid
                    config_epoch_pass["batch_size"] = configuration.valid_batch_size
                    config_epoch_pass["seq_length"] = configuration.total_seq_length
                    config_epoch_pass["input_gaussian_noise"] = configuration.input_gaussian_noise
                    config_epoch_pass["has_meta"] = configuration.has_meta
                    config_epoch_pass["mc_samples"] = number_of_MC_samples
                    config_epoch_pass["get_vars"] = [(base_pred_mean_test, "yes_mc", "pred"),
                                                     (base_pred_var_test, "yes", "pred_ale")]
                    config_epoch_pass["feed_dict"] = {batch_size_tensor: configuration.valid_batch_size,
                                                      is_training_dropout_tensor: is_epistemic,
                                                      is_training_batchnorm_tensor: False,
                                                      true_raw_test: "true_raw",
                                                      audio_test: "audio",
                                                      audio_features_test: "audio_features",
                                                      image_features_appearance_test: "image_features_appearance",
                                                      image_features_geometric_test: "image_features_geometric"}
                    if configuration.has_meta:
                        config_epoch_pass["get_vars"].append((teacher_weights_test, "yes", "informativeness"))

                        config_epoch_pass["feed_dict"][source_pred_mean_input_test] = "source_pred_mean_input"
                        config_epoch_pass["feed_dict"][source_pred_var_input_test] = "source_pred_var_input"
                        config_epoch_pass["feed_dict"][source_pred_mean_epistemic_input_test] = "source_pred_mean_epistemic_input"

                        config_epoch_pass["source_vars"] = {"pred": src_item_pred,
                                                            "pred_ale": src_item_pred_ale,
                                                            "pred_epi": src_item_pred_epi}
                        config_epoch_pass["source_subject_to_id"] = src_subject_to_id

                    config_epoch_pass["saver"] = None

                    config_epoch_pass = dict_to_struct(config_epoch_pass)

                    base_valid_items, base_valid_subject_to_id = run_epoch(config_epoch_pass)

                    config_post_processing = dict()
                    for emotion in ["arousal", "valence"]:
                        config_post_processing[emotion] = dict()
                        config_post_processing[emotion]["number_of_items"] = configuration.valid_size
                        config_post_processing[emotion]["batch_size"] = configuration.valid_batch_size
                        config_post_processing[emotion]["seq_length"] = configuration.total_seq_length
                    config_post_processing["arousal"]["true"] = base_valid_items.arousal.true
                    config_post_processing["arousal"]["pred"] = base_valid_items.arousal.pred
                    config_post_processing["valence"]["true"] = base_valid_items.valence.true
                    config_post_processing["valence"]["pred"] = base_valid_items.valence.pred
                    for emotion in ["arousal", "valence"]:
                        config_post_processing[emotion] = dict_to_struct(config_post_processing[emotion])

                    config_post_processing = dict_to_struct(config_post_processing)

                    base_valid_postprocess_struct = post_process.post_process_output(config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_valid_items.arousal.pred
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing["best_filter_size"] = base_train_postprocess_struct.arousal.best_filter_size
                    config_post_processing["was_centred"] = base_train_postprocess_struct.arousal.was_centred
                    config_post_processing["centre_offset"] = base_train_postprocess_struct.arousal.centre_offset
                    config_post_processing["was_scaled"] = base_train_postprocess_struct.arousal.was_scaled
                    config_post_processing["scale_multiplier"] = base_train_postprocess_struct.arousal.scale_multiplier
                    config_post_processing["best_time_shift"] = base_train_postprocess_struct.arousal.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    arousal_base_valid_pred_pp = post_process.post_process_specific_emotion_specific_parameters(config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_valid_items.valence.pred
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing["best_filter_size"] = base_train_postprocess_struct.valence.best_filter_size
                    config_post_processing["was_centred"] = base_train_postprocess_struct.valence.was_centred
                    config_post_processing["centre_offset"] = base_train_postprocess_struct.valence.centre_offset
                    config_post_processing["was_scaled"] = base_train_postprocess_struct.valence.was_scaled
                    config_post_processing["scale_multiplier"] = base_train_postprocess_struct.valence.scale_multiplier
                    config_post_processing["best_time_shift"] = base_train_postprocess_struct.valence.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    valence_base_valid_pred_pp = post_process.post_process_specific_emotion_specific_parameters(config_post_processing)

                    arousal_loss_valid = metrics.batch_concordance_cc_numpy(base_valid_items.arousal.true.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)),
                                                                                base_valid_items.arousal.pred.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)))
                    valence_loss_valid = metrics.batch_concordance_cc_numpy(base_valid_items.valence.true.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)),
                                                                                base_valid_items.valence.pred.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)))

                    loss_valid = (arousal_loss_valid + valence_loss_valid) / 2.0

                    # Score with post-processing
                    arousal_loss_valid_pp = metrics.batch_concordance_cc_numpy(base_valid_items.arousal.true.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)),
                                                                                   arousal_base_valid_pred_pp.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)))
                    valence_loss_valid_pp = metrics.batch_concordance_cc_numpy(base_valid_items.valence.true.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)),
                                                                                   valence_base_valid_pred_pp.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)))

                    loss_valid_pp = (arousal_loss_valid_pp + valence_loss_valid_pp) / 2.0

                    print("Test Base model.")
                    config_epoch_pass = dict()
                    config_epoch_pass["sess"] = sess
                    config_epoch_pass["init_op"] = init_op_test
                    config_epoch_pass["steps_per_epoch"] = val_steps_per_epoch
                    config_epoch_pass["next_element"] = next_element_test
                    config_epoch_pass["batch_size"] = configuration.valid_batch_size
                    config_epoch_pass["seq_length"] = configuration.total_seq_length
                    config_epoch_pass["input_gaussian_noise"] = configuration.input_gaussian_noise
                    config_epoch_pass["is_training_dropout_tensor"] = is_epistemic
                    config_epoch_pass["is_training_batchnorm_tensor"] = False
                    config_epoch_pass["has_meta"] = False
                    config_epoch_pass["mc_samples"] = number_of_MC_samples
                    config_epoch_pass["get_vars"] = [(base_pred_mean_test, "yes_mc", "pred"),
                                                     (base_pred_var_test, "yes", "pred_ale")]
                    config_epoch_pass["feed_dict"] = {batch_size_tensor: configuration.valid_batch_size,
                                                      is_training_dropout_tensor: is_epistemic,
                                                      is_training_batchnorm_tensor: False,
                                                      audio_test: "audio",
                                                      audio_features_test: "audio_features",
                                                      image_features_appearance_test: "image_features_appearance",
                                                      image_features_geometric_test: "image_features_geometric"}

                    config_epoch_pass["saver"] = None

                    config_epoch_pass = dict_to_struct(config_epoch_pass)

                    base_test_items, base_test_subject_to_id = run_epoch(config_epoch_pass)

                    # Post process test with train.
                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.arousal.pred
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing["best_filter_size"] = base_train_postprocess_struct.arousal.best_filter_size
                    config_post_processing["was_centred"] = base_train_postprocess_struct.arousal.was_centred
                    config_post_processing["centre_offset"] = base_train_postprocess_struct.arousal.centre_offset
                    config_post_processing["was_scaled"] = base_train_postprocess_struct.arousal.was_scaled
                    config_post_processing["scale_multiplier"] = base_train_postprocess_struct.arousal.scale_multiplier
                    config_post_processing["best_time_shift"] = base_train_postprocess_struct.arousal.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    arousal_base_test_pred_ppt = post_process.post_process_specific_emotion_specific_parameters(config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.arousal.pred_ale
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing["best_filter_size"] = base_train_postprocess_struct.arousal.best_filter_size
                    config_post_processing["was_centred"] = False
                    config_post_processing["centre_offset"] = 0.0
                    config_post_processing["was_scaled"] = False
                    config_post_processing["scale_multiplier"] = 1.0
                    config_post_processing["best_time_shift"] = base_train_postprocess_struct.arousal.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    arousal_base_test_pred_ale_ppt = post_process.post_process_specific_emotion_specific_parameters(config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.arousal.pred_epi
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing["best_filter_size"] = base_train_postprocess_struct.arousal.best_filter_size
                    config_post_processing["was_centred"] = False
                    config_post_processing["centre_offset"] = 0.0
                    config_post_processing["was_scaled"] = False
                    config_post_processing["scale_multiplier"] = 1.0
                    config_post_processing["best_time_shift"] = base_train_postprocess_struct.arousal.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    arousal_base_test_pred_epi_ppt = post_process.post_process_specific_emotion_specific_parameters(config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.valence.pred
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing["best_filter_size"] = base_train_postprocess_struct.valence.best_filter_size
                    config_post_processing["was_centred"] = base_train_postprocess_struct.valence.was_centred
                    config_post_processing["centre_offset"] = base_train_postprocess_struct.valence.centre_offset
                    config_post_processing["was_scaled"] = base_train_postprocess_struct.valence.was_scaled
                    config_post_processing["scale_multiplier"] = base_train_postprocess_struct.valence.scale_multiplier
                    config_post_processing["best_time_shift"] = base_train_postprocess_struct.valence.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    valence_base_test_pred_ppt = post_process.post_process_specific_emotion_specific_parameters(config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.valence.pred_ale
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing["best_filter_size"] = base_train_postprocess_struct.valence.best_filter_size
                    config_post_processing["was_centred"] = False
                    config_post_processing["centre_offset"] = 0.0
                    config_post_processing["was_scaled"] = False
                    config_post_processing["scale_multiplier"] = 1.0
                    config_post_processing["best_time_shift"] = base_train_postprocess_struct.valence.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    valence_base_test_pred_ale_ppt = post_process.post_process_specific_emotion_specific_parameters(config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.valence.pred_epi
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing["best_filter_size"] = base_train_postprocess_struct.valence.best_filter_size
                    config_post_processing["was_centred"] = False
                    config_post_processing["centre_offset"] = 0.0
                    config_post_processing["was_scaled"] = False
                    config_post_processing["scale_multiplier"] = 1.0
                    config_post_processing["best_time_shift"] = base_train_postprocess_struct.valence.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    valence_base_test_pred_epi_ppt = post_process.post_process_specific_emotion_specific_parameters(config_post_processing)

                    # Post process test with valid.
                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.arousal.pred
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing[
                            "best_filter_size"] = base_valid_postprocess_struct.arousal.best_filter_size
                    config_post_processing["was_centred"] = base_valid_postprocess_struct.arousal.was_centred
                    config_post_processing["centre_offset"] = base_valid_postprocess_struct.arousal.centre_offset
                    config_post_processing["was_scaled"] = base_valid_postprocess_struct.arousal.was_scaled
                    config_post_processing[
                            "scale_multiplier"] = base_valid_postprocess_struct.arousal.scale_multiplier
                    config_post_processing[
                            "best_time_shift"] = base_valid_postprocess_struct.arousal.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    arousal_base_test_pred_ppv = post_process.post_process_specific_emotion_specific_parameters(
                            config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.arousal.pred_ale
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing[
                            "best_filter_size"] = base_valid_postprocess_struct.arousal.best_filter_size
                    config_post_processing["was_centred"] = False
                    config_post_processing["centre_offset"] = 0.0
                    config_post_processing["was_scaled"] = False
                    config_post_processing["scale_multiplier"] = 1.0
                    config_post_processing[
                            "best_time_shift"] = base_valid_postprocess_struct.arousal.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    arousal_base_test_pred_ale_ppv = post_process.post_process_specific_emotion_specific_parameters(
                            config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.arousal.pred_epi
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing[
                            "best_filter_size"] = base_valid_postprocess_struct.arousal.best_filter_size
                    config_post_processing["was_centred"] = False
                    config_post_processing["centre_offset"] = 0.0
                    config_post_processing["was_scaled"] = False
                    config_post_processing["scale_multiplier"] = 1.0
                    config_post_processing[
                            "best_time_shift"] = base_valid_postprocess_struct.arousal.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    arousal_base_test_pred_epi_ppv = post_process.post_process_specific_emotion_specific_parameters(
                            config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.valence.pred
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing[
                            "best_filter_size"] = base_valid_postprocess_struct.valence.best_filter_size
                    config_post_processing["was_centred"] = base_valid_postprocess_struct.valence.was_centred
                    config_post_processing["centre_offset"] = base_valid_postprocess_struct.valence.centre_offset
                    config_post_processing["was_scaled"] = base_valid_postprocess_struct.valence.was_scaled
                    config_post_processing[
                            "scale_multiplier"] = base_valid_postprocess_struct.valence.scale_multiplier
                    config_post_processing[
                            "best_time_shift"] = base_valid_postprocess_struct.valence.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    valence_base_test_pred_ppv = post_process.post_process_specific_emotion_specific_parameters(
                            config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.valence.pred_ale
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing[
                            "best_filter_size"] = base_valid_postprocess_struct.valence.best_filter_size
                    config_post_processing["was_centred"] = False
                    config_post_processing["centre_offset"] = 0.0
                    config_post_processing["was_scaled"] = False
                    config_post_processing["scale_multiplier"] = 1.0
                    config_post_processing[
                            "best_time_shift"] = base_valid_postprocess_struct.valence.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    valence_base_test_pred_ale_ppv = post_process.post_process_specific_emotion_specific_parameters(
                            config_post_processing)

                    config_post_processing = dict()
                    config_post_processing["pred"] = base_test_items.valence.pred_epi
                    config_post_processing["number_of_items"] = configuration.valid_size
                    config_post_processing["batch_size"] = configuration.valid_batch_size
                    config_post_processing["seq_length"] = configuration.total_seq_length
                    config_post_processing[
                            "best_filter_size"] = base_valid_postprocess_struct.valence.best_filter_size
                    config_post_processing["was_centred"] = False
                    config_post_processing["centre_offset"] = 0.0
                    config_post_processing["was_scaled"] = False
                    config_post_processing["scale_multiplier"] = 1.0
                    config_post_processing[
                            "best_time_shift"] = base_valid_postprocess_struct.valence.best_time_shift

                    config_post_processing = dict_to_struct(config_post_processing)

                    valence_base_test_pred_epi_ppv = post_process.post_process_specific_emotion_specific_parameters(
                            config_post_processing)

                    base_test_items_ppv = dict()
                    base_test_items_ppv["arousal"] = dict()
                    base_test_items_ppv["valence"] = dict()

                    base_test_items_ppv["arousal"]["pred"] = arousal_base_test_pred_ppv.reshape((9, 7500))
                    base_test_items_ppv["valence"]["pred"] = valence_base_test_pred_ppv.reshape((9, 7500))

                    base_test_items_ppv["arousal"] = dict_to_struct(base_test_items_ppv["arousal"])
                    base_test_items_ppv["valence"] = dict_to_struct(base_test_items_ppv["valence"])
                    base_test_items_ppv = dict_to_struct(base_test_items_ppv)

                    arousal_loss_test = metrics.batch_concordance_cc_numpy(base_test_items.arousal.true.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)),
                                                                           base_test_items.arousal.pred.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)))
                    valence_loss_test = metrics.batch_concordance_cc_numpy(base_test_items.valence.true.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)),
                                                                           base_test_items.valence.pred.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)))

                    loss_test = (arousal_loss_test + valence_loss_test) / 2.0

                    # Score with post-processing with train.
                    arousal_loss_test_ppt = metrics.batch_concordance_cc_numpy(base_test_items.arousal.true.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)),
                                                                              arousal_base_test_pred_ppt.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)))
                    valence_loss_test_ppt = metrics.batch_concordance_cc_numpy(base_test_items.valence.true.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)),
                                                                              valence_base_test_pred_ppt.reshape((configuration.valid_size,
                                                                                                                       configuration.valid_batch_size * configuration.total_seq_length)))

                    loss_test_ppt = (arousal_loss_test_ppt + valence_loss_test_ppt) / 2.0

                    # Score with post-processing with valid.
                    arousal_loss_test_ppv = metrics.batch_concordance_cc_numpy(
                            base_test_items.arousal.true.reshape((configuration.valid_size,
                                                                  configuration.valid_batch_size * configuration.total_seq_length)),
                            arousal_base_test_pred_ppv.reshape((configuration.valid_size,
                                                               configuration.valid_batch_size * configuration.total_seq_length)))
                    valence_loss_test_ppv = metrics.batch_concordance_cc_numpy(
                            base_test_items.valence.true.reshape((configuration.valid_size,
                                                                  configuration.valid_batch_size * configuration.total_seq_length)),
                            valence_base_test_pred_ppv.reshape((configuration.valid_size,
                                                               configuration.valid_batch_size * configuration.total_seq_length)))

                    loss_test_ppv = (arousal_loss_test_ppv + valence_loss_test_ppv) / 2.0

                    current_performance = loss_valid_pp
                    if best_performance == 0.0:
                        best_performance = current_performance

                    if current_performance > best_performance:
                        base_saver.save(sess, base_model_path_best)
                        best_performance = current_performance
                        if configuration.has_meta:
                            teacher_saver.save(sess, teacher_model_path_best)

                        to_write = [epoch,
                                    base_train_items.loss,
                                    arousal_loss_valid,
                                    valence_loss_valid,
                                    loss_valid,
                                    arousal_loss_valid_pp,
                                    valence_loss_valid_pp,
                                    loss_valid_pp,
                                    arousal_loss_test,
                                    valence_loss_test,
                                    loss_test,
                                    arousal_loss_test_ppt,
                                    valence_loss_test_ppt,
                                    loss_test_ppt,
                                    arousal_loss_test_ppv,
                                    valence_loss_test_ppv,
                                    loss_test_ppv]
                        to_write = [repr(a) for a in to_write]
                        to_write = "\t".join(to_write) + "\n"
                        with open(results_log_file, "a") as losses_fp:
                            losses_fp.write(to_write)

                        print(epoch,
                              base_train_items.loss,
                              arousal_loss_valid_pp,
                              valence_loss_valid_pp,
                              loss_valid_pp,
                              arousal_loss_test_ppt,
                              valence_loss_test_ppt,
                              loss_test_ppt,
                              arousal_loss_test_ppv,
                              valence_loss_test_ppv,
                              loss_test_ppv)
                else:
                    print(epoch,
                          base_train_items.loss)

            losses_fp.close()
