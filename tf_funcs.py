import tensorflow as tf
import numpy as np
from common_params import DIRECTORIES, NAMING, NUMBERS


def separate_pos_neg_params(params_):
    pos_params = np.zeros(params_.shape)
    neg_params = np.zeros(params_.shape)
    pos_inds = np.where(params_ >= 0)
    neg_inds = np.where(params_ < 0)
    pos_params[pos_inds] = params_[pos_inds]
    neg_params[neg_inds] = -params_[neg_inds]
    return np.concatenate((pos_params, neg_params), axis=1)


VF_POS_NP = np.load(DIRECTORIES.vf / NAMING.npy.id_p_thetas)
VF_NEG_NP = np.load(DIRECTORIES.vf / NAMING.npy.id_m_thetas)
RANDOM_NP = np.load(DIRECTORIES.vf / NAMING.npy.random_thetas)
RANDOM_VAL_NP = np.load(DIRECTORIES.vf / NAMING.npy.random_val_thetas)
PARAMS_NP = np.load(DIRECTORIES.vf / NAMING.npy.random_params)
PARAMS_VAL_NP = np.load(DIRECTORIES.vf / NAMING.npy.random_val_params)


VF = tf.constant(np.concatenate([VF_POS_NP, VF_NEG_NP], axis=0))
G = tf.constant(np.load(DIRECTORIES.vf / NAMING.npy.generic_thetas))

RANDOM = tf.constant(RANDOM_NP)
PARAMS = tf.constant(separate_pos_neg_params(PARAMS_NP))

RANDOM_VAL = tf.constant(RANDOM_VAL_NP)
PARAMS_VAL = tf.constant(separate_pos_neg_params(PARAMS_VAL_NP))


@tf.function
def predict(visual_fields, generic, params_=PARAMS):
    return params_ @ (visual_fields - generic) + generic


@tf.function
def loss_pred(visual_fields, generic, params_=PARAMS, rand_=RANDOM):
    predicted = predict(visual_fields, generic, params_)
    return tf.reduce_sum((rand_ - predicted) ** 2)


@tf.function
def loss_orig(visual_fields, generic):
    return (tf.reduce_sum((visual_fields - VF) ** 2) +
            tf.reduce_sum((generic - G) ** 2))


@tf.function
def loss(visual_fields, generic, params_=PARAMS, rand_=RANDOM, frac_pred=0.75):
    return (frac_pred * loss_pred(visual_fields, generic, params_, rand_) +
            (1 - frac_pred) * loss_orig(visual_fields, generic))


@tf.function
def loss_val(visual_fields, generic):
    return loss_pred(visual_fields, generic, PARAMS_VAL, RANDOM_VAL) / NUMBERS.num_val


@tf.function
def any_nan(arr):
    return tf.math.reduce_any(tf.math.is_nan(arr))
