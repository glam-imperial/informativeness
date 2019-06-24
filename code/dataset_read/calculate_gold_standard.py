from itertools import combinations

import numpy as np
from numpy.matlib import repmat
import pandas as pd


def get_centred_gold_standard(output_folder, ratings_folder, subject_set):
    ####################################################################################################################
    # Constants.
    ####################################################################################################################
    emotional_dimensions = 2
    emotional_dimension_names = ["arousal",
                                 "valence"]
    number_of_raters = 6
    number_of_frames_per_rating = 7501

    number_of_files = len(subject_set)

    frame_time = np.arange(0, 300.04, 0.04, dtype=np.float32)

    all_ratings = dict()
    for f, f_name in enumerate(subject_set):
        all_ratings[f_name] = dict()
        for d, d_name in enumerate(emotional_dimension_names):
            # file_path = ratings_folder + "/" + d_name + "/P" + repr(f_name) + ".csv"
            file_path = ratings_folder + "/" + d_name + "/" + f_name + ".csv"

            all_ratings[f_name][d_name] = read_ratings(file_path)

    ####################################################################################################################
    # Compute inter-rater agreement with CCC centring
    ####################################################################################################################
    rater_cnk_list = [cnk for cnk in combinations(range(number_of_raters), 2)]
    number_of_combinations = len(rater_cnk_list)
    rater_cnk = np.empty((number_of_combinations, 2), dtype=np.int32)
    for i, cnk in enumerate(rater_cnk_list):
        rater_cnk[i, :] = cnk

    rater_appearance = np.ones((number_of_raters - 1, number_of_raters), dtype=np.int32) * -1
    for r in range(number_of_raters):
        rater_appearance[:, r] = np.concatenate([np.where(rater_cnk[:, 0] == r)[0], np.where(rater_cnk[:, 1] == r)[0]])

    RMSE_raw = np.zeros((1, emotional_dimensions))
    CC_raw = np.zeros((1, emotional_dimensions))
    CCC_raw = np.zeros((1, emotional_dimensions))
    # ICC_raw = np.zeros((1, emotional_dimensions))
    alpha_raw = np.zeros((1, emotional_dimensions))
    for d, d_name in enumerate(emotional_dimension_names):
        RMSE_raw[0, d], CC_raw[0, d], CCC_raw[0, d], alpha_raw[0, d] = raters_agreement(all_ratings, d_name)
        print(RMSE_raw[0, d])
        print(CC_raw[0, d])
        print(CCC_raw[0, d])
        print(alpha_raw[0, d])

    all_ratings_CCC_cent = dict()

    for f, f_name in enumerate(subject_set):
        all_ratings_CCC_cent[f_name] = dict()
        for d, d_name in enumerate(emotional_dimension_names):

            ratings = all_ratings[f_name][d_name]

            CCC = np.zeros((1, number_of_combinations), dtype=np.float32)
            for k in range(number_of_combinations):
                CCC[0, k] = CCC_calc(ratings[:, rater_cnk[k, 0]],
                                     ratings[:, rater_cnk[k, 1]])

            CCC_agr = np.zeros((1, number_of_raters), dtype=np.float32)
            for r in range(number_of_raters):
                CCC_agr[0, r] = np.mean(CCC[0, rater_appearance[:, r]])

            mean_rating = np.mean(ratings, 0).reshape((1, number_of_raters))
            wgh_ref_CCC = np.sum(np.multiply(mean_rating, CCC_agr)/np.sum(CCC_agr))

            all_ratings_CCC_cent[f_name][d_name] = all_ratings[f_name][d_name] - repmat(mean_rating.transpose(), 1, number_of_frames_per_rating).transpose() + wgh_ref_CCC
            # output_file_path = output_folder + "/" + d_name + "/P" + repr(f_name) + ".csv"
            output_file_path = output_folder + "/" + d_name + "/" + f_name + ".csv"
            write_ratings(output_file_path, all_ratings_CCC_cent[f_name][d_name])

            output_file_path = output_folder + "/" + d_name + "/gs_" + f_name + ".csv"
            write_ratings_gs(output_file_path, all_ratings_CCC_cent[f_name][d_name])

    RMSE_CCC_cent = np.zeros((1, emotional_dimensions))
    CC_CCC_cent = np.zeros((1, emotional_dimensions))
    CCC_CCC_cent = np.zeros((1, emotional_dimensions))
    # ICC_CCC_cent = np.zeros((1, emotional_dimensions))
    alpha_CCC_cent = np.zeros((1, emotional_dimensions))
    print()
    for d, d_name in enumerate(emotional_dimension_names):
        RMSE_CCC_cent[0, d], CC_CCC_cent[0, d], CCC_CCC_cent[0, d], alpha_CCC_cent[0, d] = raters_agreement(all_ratings_CCC_cent, d_name)
        print(RMSE_CCC_cent[0, d])
        print(CC_CCC_cent[0, d])
        print(CCC_CCC_cent[0, d])
        print(alpha_CCC_cent[0, d])


def read_ratings(file_path):
    rating_columns = ["FM1 ",
                      "FM2 ",
                      "FM3 ",
                      "FF1 ",
                      "FF2 ",
                      "FF3"]

    data_frame = pd.read_csv(file_path, delimiter=';')
    array = data_frame[rating_columns].values

    return array


def write_ratings(file_path, array):
    rating_columns = ["FM1 ",
                      "FM2 ",
                      "FM3 ",
                      "FF1 ",
                      "FF2 ",
                      "FF3"]

    data_frame = pd.DataFrame(data=array, columns=rating_columns, dtype=np.float32)
    data_frame.to_csv(path_or_buf=file_path, sep=";")


def write_ratings_gs(file_path, array):
    rating_columns = ["GS"]

    data_frame = pd.DataFrame(data=array.mean(axis=1), columns=rating_columns, dtype=np.float32)
    data_frame.to_csv(path_or_buf=file_path, sep=";")


def raters_agreement(ratings_dict, d_name):
    ratings = np.vstack([ratings_dict[f_name][d_name] for f_name in ratings_dict.keys()])
    # print(ratings)
    # print(ratings.shape)

    number_of_raters = ratings.shape[1]

    rater_cnk_list = [cnk for cnk in combinations(range(number_of_raters), 2)]
    number_of_combinations = len(rater_cnk_list)
    rater_cnk = np.empty((number_of_combinations, 2), dtype=np.int32)
    for i, cnk in enumerate(rater_cnk_list):
        rater_cnk[i, :] = cnk

    RMSE = np.zeros((1, number_of_combinations), dtype=np.float32)
    CC = np.zeros((1, number_of_combinations), dtype=np.float32)
    CCC = np.zeros((1, number_of_combinations), dtype=np.float32)
    for k in range(number_of_combinations):
        RMSE[0, k], CC[0, k], CCC[0, k] = raters_statistics(ratings[:, rater_cnk[k, 0]],
                                                            ratings[:, rater_cnk[k, 1]])

    # ICC = ICC_shrout(3, "k", ratings.transpose())

    alpha = cronbach(ratings.T)

    RMSE = np.mean(RMSE)
    CC = np.mean(CC)
    CCC = np.mean(CCC)

    return RMSE, CC, CCC, alpha


def raters_statistics(r1, r2):
    MSE = np.nanmean(np.power(r1-r2, 2))
    RMSE = np.sqrt(MSE)

    r1_mean = np.nanmean(r1)
    r2_mean = np.nanmean(r2)
    r1_std = np.nanstd(r1)
    r2_std = np.nanstd(r2)
    covariance = np.nanmean(np.multiply(r1-r1_mean, r2-r2_mean))
    CC = covariance / (r1_std * r2_std)

    r1_var = np.power(r1_std, 2)
    r2_var = np.power(r2_std, 2)
    CCC = (2 * covariance) / (r1_var + r2_var + np.power(r1_mean - r2_mean, 2))

    return RMSE, CC, CCC


def cronbach(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems - 1.) * (1 - itemvars.sum() / tscores.var(ddof=1))


def CCC_calc(r1, r2):
    r1_mean = np.nanmean(r1)
    r2_mean = np.nanmean(r2)
    r1_std = np.nanstd(r1)
    r2_std = np.nanstd(r2)
    covariance = np.nanmean(np.multiply(r1 - r1_mean, r2 - r2_mean))

    r1_var = np.power(r1_std, 2)
    r2_var = np.power(r2_std, 2)
    CCC = (2 * covariance) / (r1_var + r2_var + np.power(r1_mean - r2_mean, 2))

    return CCC


if __name__ == "__main__":
    ratings_folder = "/path/to/AVEC2016/ratings_individual"
    all_subject_set = ["train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_7", "train_8", "train_9",
                       "dev_1", "dev_2", "dev_3", "dev_4", "dev_5", "dev_6", "dev_7", "dev_8", "dev_9"]

    output_folder = "/path/to/AVEC2016/ratings_individual_centred"
    get_centred_gold_standard(output_folder, ratings_folder, all_subject_set)

