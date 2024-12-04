import tensorflow as tf
import numpy as np
from common_params import DIRECTORIES, NUMBERS
from naming import NAMING


def separate_pos_neg_params(params_):
    """Create separate arrays for positive and negative parameters.

    `params_` is a 2D array with shape (number of faces, number of parameters).
    First, two separate arrays with the same shape as `params_` are created.
    They are named `pos_params` and `neg_params` and set to 0.
    Then, for each element in `params_`, if it is positive,
    pos_params[corresponding index] is set to the element.
    If it is negative, neg_params[corresponding index] is set to
    the negative of the element.
    Finally, the two arrays are concatenated along the second axis and returned.


    Parameters
    ----------
    params_ : numpy.ndarray
        2D array with shape (number of faces, number of parameters).

    Returns
    -------
    numpy.ndarray
        2D array with shape (number of faces, 2 * number of parameters).
    """
    pos_params = np.zeros(params_.shape)
    neg_params = np.zeros(params_.shape)
    pos_inds = np.where(params_ >= 0)
    neg_inds = np.where(params_ < 0)
    pos_params[pos_inds] = params_[pos_inds]
    neg_params[neg_inds] = -params_[neg_inds]
    return np.concatenate((pos_params, neg_params), axis=1)


VF_POS_NP = np.load(DIRECTORIES.vf / NAMING.id.pos.theta_boundary.npy)
VF_NEG_NP = np.load(DIRECTORIES.vf / NAMING.id.neg.theta_boundary.npy)
RANDOM_NP = np.load(DIRECTORIES.vf / NAMING.random.theta_boundary.npy)
RANDOM_VAL_NP = np.load(DIRECTORIES.vf / NAMING.random.val.theta_boundary.npy)
PARAMS_NP = np.load(DIRECTORIES.vf / NAMING.random.params.npy)
PARAMS_VAL_NP = np.load(DIRECTORIES.vf / NAMING.random.val.params.npy)


VF = tf.constant(np.concatenate([VF_POS_NP, VF_NEG_NP], axis=0))
G = tf.constant(np.load(DIRECTORIES.vf / NAMING.generic.theta_boundary.npy))

RANDOM = tf.constant(RANDOM_NP)
PARAMS = tf.constant(separate_pos_neg_params(PARAMS_NP))

RANDOM_VAL = tf.constant(RANDOM_VAL_NP)
PARAMS_VAL = tf.constant(separate_pos_neg_params(PARAMS_VAL_NP))


@tf.function
def predict(visual_fields, generic, params_=PARAMS):
    """Predict the Visual Field boundaries for faces given by `params_`.

    Parameters
    ----------
    visual_fields : tf.Tensor | numpy.ndarray
        The VF boundaries corresponding to the id parameters.
    generic : tf.Tensor | numpy.ndarray
        The VF boundary corresponding to the generic mesh.
    params_ : tf.Tensor | numpy.ndarray
        The parameters of the faces for which the VF boundaries
        are to be predicted.
        The shape is (number of faces, 2 * number of parameters) because
        the parameters are separated into positive and negative parts.

    Returns
    -------
    tf.Tensor
        The predicted VF boundaries for the faces.
    """
    return params_ @ (visual_fields - generic) + generic


@tf.function
def loss_pred(visual_fields, generic, params_=PARAMS, rand_=RANDOM):
    """Calculate the squared error for the predicted Visual Fields.

    Parameters
    ----------
    visual_fields : tf.Tensor | numpy.ndarray
        The VF boundaries corresponding to the id parameters.
    generic : tf.Tensor | numpy.ndarray
        The VF boundary corresponding to the generic mesh.
    params_ : tf.Tensor | numpy.ndarray
        The parameters of the faces for which the VF boundaries
        are to be predicted.
        The shape is (number of faces, 2 * number of parameters) because
        the parameters are separated into positive and negative parts.
    rand_ : tf.Tensor | numpy.ndarray
        The rendered VF boundaries for the faces given by `params_`.

    Returns
    -------
    tf.Tensor
        The squared error of the predicted VF boundaries from the rendered ones.
    """
    predicted = predict(visual_fields, generic, params_)
    return tf.reduce_sum((rand_ - predicted) ** 2)


@tf.function
def loss_orig(visual_fields, generic):
    """Calculate the squared error of the optimized VF boundaries for the
    id and generic faces from the rendered ones.

    Parameters
    ----------
    visual_fields : tf.Tensor | numpy.ndarray
        The optimized VF boundaries corresponding to the id parameters.
    generic : tf.Tensor | numpy.ndarray
        The optimized VF boundary corresponding to the generic mesh.

    Returns
    -------
    tf.Tensor
        The squared error of the optimized VF boundaries from the rendered ones.
    """
    return (tf.reduce_sum((visual_fields - VF) ** 2) +
            tf.reduce_sum((generic - G) ** 2))


@tf.function
def loss(visual_fields, generic, params_=PARAMS, rand_=RANDOM, frac_pred=0.75):
    """Calculate the loss for VF boundary optimization.

    The loss is a weighted sum of the loss for the predicted VF boundaries
    for random faces and the loss for the optimized id and generic VF boundaries
    from the rendered values.

    Parameters
    ----------
    visual_fields : tf.Tensor | numpy.ndarray
        The optimized VF boundaries corresponding to the id parameters.
    generic : tf.Tensor | numpy.ndarray
        The optimized VF boundary corresponding to the generic mesh.
    params_ : tf.Tensor | numpy.ndarray
        The parameters of the faces for which the VF boundaries
        are to be predicted.
        The shape is (number of faces, 2 * number of parameters) because
        the parameters are separated into positive and negative parts.
    rand_ : tf.Tensor | numpy.ndarray
        The rendered VF boundaries for the faces given by `params_`.
    frac_pred : float, default 0.75
        The weight for the loss of the predicted VF boundaries for random faces.
        The weight for the loss of the optimized VF boundaries is `1 - frac_pred`.

    Returns
    -------
    tf.Tensor
        The loss for the VF boundary optimization.
        `frac_pred` * `loss_pred` + (1 - `frac_pred`) * `loss_orig`
    """
    return (frac_pred * loss_pred(visual_fields, generic, params_, rand_) +
            (1 - frac_pred) * loss_orig(visual_fields, generic))


@tf.function
def loss_val(visual_fields, generic):
    """Calculate the loss for the validation set.

    Parameters
    ----------
    visual_fields : tf.Tensor | numpy.ndarray
        The VF boundaries corresponding to the id parameters.
    generic : tf.Tensor | numpy.ndarray
        The VF boundary corresponding to the generic mesh.

    Returns
    -------
    tf.Tensor
        The loss for the validation set. Calculated using `loss_pred`.
        Divided by the number of faces in the validation set.
    """
    return loss_pred(visual_fields, generic, PARAMS_VAL, RANDOM_VAL) / NUMBERS.num_val


@tf.function
def any_nan(arr):
    """Check if there are any NaN values in the array.

    Parameters
    ----------
    arr : tf.Tensor | numpy.ndarray
        The array to check for NaN values.

    Returns
    -------
    tf.Tensor
        True if there are any NaN values in the array, False otherwise.
        To get the boolean value, use `.numpy()`.
    """
    return tf.math.reduce_any(tf.math.is_nan(arr))
