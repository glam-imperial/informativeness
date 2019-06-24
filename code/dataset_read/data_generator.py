from io import BytesIO
from pathlib import Path

import menpo
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.ndimage.filters import generic_filter
from moviepy.editor import VideoFileClip
from menpo.visualize import print_progress

########################################################################################################################
# Preprocess either train-valid or test.
########################################################################################################################
ROOT_STR = "/path/to/AVEC2016"
TRUE_FOLDER = "/path/to/test_data"
TARGET_STR = "/path/to/preprocessed_data"

portion_to_id = dict(
    train = [1, 2, 3, 4, 5, 6, 7, 8, 9],
    valid = [1, 2, 3, 4, 5, 6, 7, 8, 9]
)

# portion_to_id = dict(
#     test = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# )

########################################################################################################################
########################################################################################################################

root_dir = Path(ROOT_STR)

MAPPING = {
    'test_6': 'P13',
    'test_9': 'P20',
    'test_2': 'P32',
    'test_1': 'P38',
    'test_7': 'P47',
    'test_8': 'P49',
    'test_3': 'P53',
    'test_5': 'P59',
    'test_4': 'P63'
}


def read_csv(path):
    data = list()
    with open(path, "r") as fp:
        for line in fp:
            clean_line = line.strip().split(",")
            data.append(float(clean_line[1]))
    data = np.array(data, dtype=np.float32)
    return data


def get_true(folder, mapping):
    data = dict()
    data["arousal"] = list()
    data["valence"] = list()

    for i in range(1, 10):
        file_name = mapping["test_" + repr(i)]

        data["arousal"].append(read_csv(folder + "/arousal/" + file_name + ".csv"))
        data["valence"].append(read_csv(folder + "/valence/" + file_name + ".csv"))
    data["arousal"] = np.vstack(data["arousal"])
    data["valence"] = np.vstack(data["valence"])

    return data


def get_samples(subject_id, portion):
    clip = VideoFileClip(str(root_dir / "recordings_video/{}.mp4".format(subject_id)))

    subsampled_audio = clip.audio.set_fps(16000)

    # Get audio, image recordings data.
    audio_frames = []
    # image_frames = []
    for i in range(1, 7501):
        time = 0.04 * i

        audio = np.array(list(subsampled_audio.subclip(time - 0.04, time).iter_frames()))
        audio = audio.mean(1)[:640]

        audio_frames.append(audio.astype(np.float32))

        image = np.array(list(clip.subclip(time - 0.04, time).iter_frames()))[0]

    # Standardize raw audio.
    sum_value = 0.0
    sum_squares_value = 0.0
    for au in audio_frames:
        sum_value += np.sum(au)
    mean_value = sum_value / (len(audio_frames) * 640)

    for au in audio_frames:
        sum_squares_value += np.sum(np.power(au - mean_value, 2.0))
    standard_deviation = np.sqrt(sum_squares_value / (len(audio_frames) * 640))

    for i, au in enumerate(audio_frames):
        audio_frames[i] = (au - mean_value) / standard_deviation

    # Get audio, image engineered features.
    audio_features = read_avec2016_features(str(ROOT_STR + "/features_audio"), subject_id)
    image_appearance_features = read_avec2016_features(str(ROOT_STR + "/features_video_appearance"), subject_id)
    image_geometric_features = read_avec2016_features(str(ROOT_STR + "/features_video_geometric"), subject_id)

    # Get gold standard and individual ratings.
    arousal_label_path = root_dir / 'ratings_individual_centred/arousal/{}.csv'.format(subject_id)
    valence_label_path = root_dir / 'ratings_individual_centred/valence/{}.csv'.format(subject_id)
    rating_columns = ["FM1 ",
                      "FM2 ",
                      "FM3 ",
                      "FF1 ",
                      "FF2 ",
                      "FF3"]
    arousal = pd.read_csv(str(arousal_label_path), delimiter=';')
    valence = pd.read_csv(str(valence_label_path), delimiter=';')

    arousal = arousal[rating_columns].values
    valence = valence[rating_columns].values

    if portion == "test":
        true = get_true(TRUE_FOLDER, MAPPING)
        gs_arousal = true["arousal"][int(subject_id[-1])-1].reshape((7501, 1))
        gs_valence = true["valence"][int(subject_id[-1])-1].reshape((7501, 1))

        return audio_frames,\
               audio_features,\
               image_appearance_features,\
               image_geometric_features, \
               np.hstack([gs_arousal, gs_valence]).astype(np.float32)
    else:
        return audio_frames, \
               audio_features, \
               image_appearance_features, \
               image_geometric_features, \
               np.hstack([arousal, valence]).astype(np.float32)


def loadarffstr(fp):
    result = list()
    for line in fp:
        clean_line = line.strip().split(",")
        if len(clean_line) == 3:
            result.append(float(clean_line[2]))

    result_array = np.ones((len(result), 1), dtype=np.float32)
    for i, r in enumerate(result):
        result_array[i, 0] = r

    return result_array


def read_avec2016_features(path_prefix, subject_id):
    features = list()
    for target in ["arousal", "valence"]:
        with open(path_prefix + "/" + target + "/" + "{}.arff".format(subject_id), "r") as fp:
            data, meta = arff.loadarff(fp)
            data = pd.DataFrame(data)
            data = data.values

            # Standardize features.
            data_mean = np.mean(data, axis=0).reshape((1, data.shape[1]))
            data_std = np.std(data, axis=0).reshape((1, data.shape[1]))
            data = (data - data_mean) / data_std

            extra_data_mean = generic_filter(data, np.mean, size=(150, 1))
            extra_data_std = generic_filter(data, np.std, size=(150, 1))

            data = np.hstack([data, extra_data_mean, extra_data_std])

            features.append(data)
    features = np.hstack(features).astype(np.float32)

    return features


def get_jpg_string(im):
    # Gets the serialized jpg from a menpo `Image`.
    fp = BytesIO()
    menpo.io.export_image(im, fp, extension='jpg')
    fp.seek(0)
    return fp.read()

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_sample(writer, subject_id, portion):
    if portion == "train":
        subject_name = 'train_{}'.format(subject_id)
    elif portion == "valid":
        subject_name = 'dev_{}'.format(subject_id)
    elif portion == "test":
        subject_name = 'test_{}'.format(subject_id)
    else:
        raise ValueError("Invalid portion name.")

    for i, (audio,
            audio_features,
            image_features_appearance,
            image_features_geometric,
            label) in enumerate(zip(*get_samples(subject_name, portion))):
        label_shape = np.array(label.shape)
        # gs_label_shape = np.array(gs_label.shape)
        audio_features_shape = np.array(audio_features.shape)
        image_features_appearance_shape = np.array(image_features_appearance.shape)
        image_features_geometric_shape = np.array(image_features_geometric.shape)
        # image_shape = np.array(image.shape)
        raw_audio_shape = np.array(audio.shape)
        # print(label_shape, gs_label_shape, audio_features_shape, image_features_appearance_shape, image_features_geometric_shape)

        if portion == "test":
            example = tf.train.Example(features=tf.train.Features(feature={
                        'sample_id': _int_feature(i),
                        'subject_id': _int_feature(subject_id),
                        'gs_label': _bytes_feature(label.tobytes()),
                        'gs_label_shape': _bytes_feature(label_shape.tobytes()),
                        'raw_audio': _bytes_feature(audio.tobytes()),
                        'raw_audio_shape': _bytes_feature(raw_audio_shape.tobytes()),
                        'audio_features': _bytes_feature(audio_features.tobytes()),
                        'audio_features_shape': _bytes_feature(audio_features_shape.tobytes()),
                        'image_features_appearance': _bytes_feature(image_features_appearance.tobytes()),
                        'image_features_appearance_shape': _bytes_feature(image_features_appearance_shape.tobytes()),
                        'image_features_geometric': _bytes_feature(image_features_geometric.tobytes()),
                        'image_features_geometric_shape': _bytes_feature(image_features_geometric_shape.tobytes()),
                    }))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'sample_id': _int_feature(i),
                'subject_id': _int_feature(subject_id),
                'label': _bytes_feature(label.tobytes()),
                'label_shape': _bytes_feature(label_shape.tobytes()),
                'raw_audio': _bytes_feature(audio.tobytes()),
                'raw_audio_shape': _bytes_feature(raw_audio_shape.tobytes()),
                'audio_features': _bytes_feature(audio_features.tobytes()),
                'audio_features_shape': _bytes_feature(audio_features_shape.tobytes()),
                'image_features_appearance': _bytes_feature(image_features_appearance.tobytes()),
                'image_features_appearance_shape': _bytes_feature(image_features_appearance_shape.tobytes()),
                'image_features_geometric': _bytes_feature(image_features_geometric.tobytes()),
                'image_features_geometric_shape': _bytes_feature(image_features_geometric_shape.tobytes()),
            }))

        writer.write(example.SerializeToString())
        del audio


def main(directory):
  for portion in portion_to_id.keys():
    print(portion)

    for subj_id in print_progress(portion_to_id[portion]):

      writer = tf.python_io.TFRecordWriter(
          (directory / 'tf_records' / portion / '{}.tfrecords'.format(subj_id)
          ).as_posix())
      serialize_sample(writer, subj_id, portion)


if __name__ == "__main__":
    main(Path(TARGET_STR))
