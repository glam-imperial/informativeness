import tensorflow as tf


def concordance_cc(pred, true):
    pred_mean = tf.reduce_mean(pred)
    gt_mean = tf.reduce_mean(true)

    pred_var = tf.reduce_mean(tf.square(pred)) - tf.square(pred_mean)
    gt_var = tf.reduce_mean(tf.square(true)) - tf.square(gt_mean)

    mean_cent_prod = tf.reduce_mean((pred - pred_mean) * (true - gt_mean))

    return 1.0 - (2.0 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))


def aleatory_attenuated_concordance_cc(pred, log_var_pred, true):
    precision = tf.exp(-log_var_pred)

    mu_x = weighted_mean(pred, precision)
    mu_y = weighted_mean(true, precision)

    mean_cent_prod = weighted_covariance(pred, true, mu_x, mu_y, precision)
    denom = weighted_covariance(pred, pred, mu_x, mu_x, precision) +\
            weighted_covariance(true, true, mu_y, mu_y, precision) +\
            tf.square((mu_x - mu_y))

    return 1.0 - (2.0 * mean_cent_prod) / denom


########################################################################################################################
# Helper functions
########################################################################################################################

def weighted_mean(x, w):
    mu = tf.reduce_sum(tf.multiply(x, w)) / tf.reduce_sum(w)
    return mu


def weighted_mean_vector(x_vector, w):
    mu = tf.reduce_sum(tf.multiply(x_vector, w), axis=0, keep_dims=True) / tf.reduce_sum(w)
    return mu


def weighted_covariance(x, y, mu_x, mu_y, w):
    sigma = tf.reduce_sum(tf.multiply(w, tf.multiply(x - mu_x, y - mu_y))) / tf.reduce_sum(w)
    return sigma


def weighted_covariance_one_vector(x, y_vector, mu_x, mu_y_vector, w):
    sigma = tf.reduce_sum(tf.multiply(w, tf.multiply(x - mu_x, y_vector - mu_y_vector)), axis=0, keep_dims=True) / tf.reduce_sum(w)
    return sigma


def weighted_covariance_two_vectors(x_vector, y_vector, mu_x_vector, mu_y_vector, w):
    sigma = tf.reduce_sum(tf.multiply(w, tf.multiply(x_vector - mu_x_vector, y_vector - mu_y_vector)), axis=0, keep_dims=True) / tf.reduce_sum(w)
    return sigma


def get_loss_function(uncertainty):
    if uncertainty in ["aleatory", "both"]:
        loss_function = aleatory_attenuated_concordance_cc
    elif uncertainty in ["none", "epistemic"]:
        loss_function = concordance_cc
    else:
        raise ValueError("Invalid uncertainty type.")

    return loss_function
