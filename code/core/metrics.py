import numpy as np


def batch_concordance_cc_numpy(pred, true):
    pred_mean = np.mean(pred, axis=1).reshape((pred.shape[0], 1))
    pred_var = np.var(pred, axis=1).reshape((pred.shape[0], 1))

    gt_mean = np.mean(true, axis=1).reshape((true.shape[0], 1))
    gt_var = np.var(true, axis=1).reshape((true.shape[0], 1))

    mean_cent_prod = np.mean(np.multiply((pred - pred_mean), (true - gt_mean)), axis=1).reshape((pred.shape[0], 1))

    ccc = np.divide((2 * mean_cent_prod), (pred_var + gt_var + np.power(pred_mean - gt_mean, 2)))

    return np.mean(ccc, axis=0)[0]
