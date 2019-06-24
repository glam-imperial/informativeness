from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path

import tensorflow as tf


def get_split(dataset_dir,
              is_training,
              split_name,
              batch_size,
              seq_length,
              buffer_size,
              subset=None):
    root_path = Path(dataset_dir) / split_name
    if subset is None:
        paths = [str(x) for x in root_path.glob('*.tfrecords')]
    else:
        paths = [str(x) for x in root_path.glob('*.tfrecords')]

    if split_name == "test":
        dataset = tf.data.TFRecordDataset(paths)
        dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                features={
                                                                    'sample_id': tf.FixedLenFeature([], tf.int64),
                                                                    'subject_id': tf.FixedLenFeature([], tf.int64),
                                                                    'gs_label': tf.FixedLenFeature([], tf.string),
                                                                    'gs_label_shape': tf.FixedLenFeature([], tf.string),
                                                                    'raw_audio': tf.FixedLenFeature([], tf.string),
                                                                    'raw_audio_shape': tf.FixedLenFeature([],
                                                                                                          tf.string),
                                                                    'audio_features': tf.FixedLenFeature([], tf.string),
                                                                    'audio_features_shape': tf.FixedLenFeature([],
                                                                                                               tf.string),
                                                                    'image_features_appearance': tf.FixedLenFeature([],
                                                                                                                    tf.string),
                                                                    'image_features_appearance_shape': tf.FixedLenFeature(
                                                                        [], tf.string),
                                                                    'image_features_geometric': tf.FixedLenFeature([],
                                                                                                                   tf.string),
                                                                    'image_features_geometric_shape': tf.FixedLenFeature(
                                                                        [], tf.string),
                                                                }
                                                                ))

        dataset = dataset.map(lambda x: {'sample_id': tf.reshape(x['sample_id'], (1,)),
                                         'subject_id': tf.reshape(x['subject_id'], (1,)),
                                         'gs_label': tf.reshape(tf.decode_raw(x['gs_label'], tf.float32), (2,)),
                                         'gs_label_shape': tf.reshape(tf.decode_raw(x['gs_label_shape'], tf.float32), (2,)),
                                         'raw_audio': tf.reshape(tf.decode_raw(x['raw_audio'], tf.float32), (640,)),
                                         'raw_audio_shape': tf.reshape(tf.decode_raw(x['raw_audio_shape'], tf.float32),
                                                                       (2,)),
                                         'audio_features': tf.reshape(tf.decode_raw(x['audio_features'], tf.float32),
                                                                      (534,)),  # 178, 534
                                         'audio_features_shape': tf.reshape(
                                             tf.decode_raw(x['audio_features_shape'], tf.float32),
                                             (2,)),
                                         'image_features_appearance': tf.reshape(
                                             tf.decode_raw(x['image_features_appearance'], tf.float32),
                                             (1014,)),  # 338, 1014
                                         'image_features_appearance_shape': tf.reshape(
                                             tf.decode_raw(x['image_features_appearance_shape'], tf.float32),
                                             (2,)),
                                         'image_features_geometric': tf.reshape(
                                             tf.decode_raw(x['image_features_geometric'], tf.float32),
                                             (3798,)),  # 1266, 3798
                                         'image_features_geometric_shape': tf.reshape(
                                             tf.decode_raw(x['image_features_geometric_shape'], tf.float32),
                                             (2,))})
    else:
        dataset = tf.data.TFRecordDataset(paths)
        dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                features={
                                                                    'sample_id': tf.FixedLenFeature([], tf.int64),
                                                                    'subject_id': tf.FixedLenFeature([], tf.int64),
                                                                    'label': tf.FixedLenFeature([], tf.string),
                                                                    'label_shape': tf.FixedLenFeature([], tf.string),
                                                                    'raw_audio': tf.FixedLenFeature([], tf.string),
                                                                    'raw_audio_shape': tf.FixedLenFeature([],
                                                                                                          tf.string),
                                                                    'audio_features': tf.FixedLenFeature([], tf.string),
                                                                    'audio_features_shape': tf.FixedLenFeature([],
                                                                                                               tf.string),
                                                                    'image_features_appearance': tf.FixedLenFeature([],
                                                                                                                    tf.string),
                                                                    'image_features_appearance_shape': tf.FixedLenFeature(
                                                                        [], tf.string),
                                                                    'image_features_geometric': tf.FixedLenFeature([],
                                                                                                                   tf.string),
                                                                    'image_features_geometric_shape': tf.FixedLenFeature(
                                                                        [], tf.string),
                                                                }
                                                                ))

        dataset = dataset.map(lambda x: {'sample_id': tf.reshape(x['sample_id'], (1,)),
                                         'subject_id': tf.reshape(x['subject_id'], (1,)),
                                         'label': tf.reshape(tf.decode_raw(x['label'], tf.float32), (12,)),
                                         'label_shape': tf.reshape(tf.decode_raw(x['label_shape'], tf.float32), (2,)),
                                         'raw_audio': tf.reshape(tf.decode_raw(x['raw_audio'], tf.float32), (640,)),
                                         'raw_audio_shape': tf.reshape(tf.decode_raw(x['raw_audio_shape'], tf.float32),
                                                                       (2,)),
                                         'audio_features': tf.reshape(tf.decode_raw(x['audio_features'], tf.float32),
                                                                      (534,)),  # 178, 534
                                         'audio_features_shape': tf.reshape(
                                             tf.decode_raw(x['audio_features_shape'], tf.float32),
                                             (2,)),
                                         'image_features_appearance': tf.reshape(
                                             tf.decode_raw(x['image_features_appearance'], tf.float32),
                                             (1014,)),  # 338, 1014
                                         'image_features_appearance_shape': tf.reshape(
                                             tf.decode_raw(x['image_features_appearance_shape'], tf.float32),
                                             (2,)),
                                         'image_features_geometric': tf.reshape(
                                             tf.decode_raw(x['image_features_geometric'], tf.float32),
                                             (3798,)),  # 1266, 3798
                                         'image_features_geometric_shape': tf.reshape(
                                             tf.decode_raw(x['image_features_geometric_shape'], tf.float32),
                                             (2,))})

    dataset = dataset.repeat()
    dataset = dataset.batch(seq_length)
    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset
