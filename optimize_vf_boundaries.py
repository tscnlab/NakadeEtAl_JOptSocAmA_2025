import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tqdm import tqdm

from common_params import DIRECTORIES, NUMBERS, save_npy_files
from naming import NAMING
from tf_funcs import predict, loss, loss_val, any_nan, VF, G, PARAMS_VAL


def gradient_step(vf_var_, g_var_, lr_):
    """Perform a gradient descent step.

    Parameters
    ----------
    vf_var_ : tf.Variable
        The variable containing the visual field boundaries for id parameters.
    g_var_ : tf.Variable
        The variable containing the generic visual field boundary.
    lr_ : float
        The learning rate.

    Returns
    -------
    tuple[float, float]
        The loss and validation loss values at the step.
    """
    with tf.GradientTape() as tape:
        loss_temp_ = loss(vf_var_, g_var_)
        dl_dvf_, dl_dg_ = tape.gradient(loss_temp_, [vf_var_, g_var_])
        _ = vf_var_.assign_sub(lr_ * dl_dvf_)
        _ = g_var_.assign_sub(lr_ * dl_dg_)
    loss_val_temp_ = loss_val(vf_var_, g_var_)
    return loss_temp_.numpy(), loss_val_temp_.numpy()


def monotonically_increasing(losses_record_, n_):
    """Check if the last :py:attr:`n_` losses are monotonically increasing.

    Parameters
    ----------
    losses_record_ : TrainLossesRecord
        The record of losses.
    n_ : int
        The number of losses to check.

    Returns
    -------
    bool
        ``True`` if the last :py:attr:`n_` losses are monotonically increasing,
        ``False`` otherwise.
    """
    return np.all(np.diff(losses_record_.losses[-n_:]) >= 0)


class LowestValLoss:
    """Record the Visual Fields corresponding to the lowest validation loss.

    Attributes
    ----------
    val_loss : float
        The lowest validation loss yet.
    iteration : int
        The iteration at which the lowest validation loss occurred.
    vf : np.ndarray
        The id Visual Fields corresponding to the lowest validation loss.
    g : np.ndarray
        The generic Visual Field corresponding to the lowest validation loss.

    Methods
    -------
    record(val_loss_, vf_var_, g_var_, iteration_)
        Update the attributes if the new :py:attr:`val_loss_` is lower than the
        recorded :py:attr:`val_loss`.
    """
    def __init__(self):
        self.val_loss = float('inf')
        self.iteration = 0
        self.vf = None
        self.g = None

    def record(self, val_loss_, vf_var_, g_var_, iteration_):
        """Update the attributes if the :py:attr:`val_loss_` is lower than
        :py:attr:`val_loss`

        During training, the new val loss is compared to the stored val loss.
        If the new val loss is lower, the attributes are updated.

        Parameters
        ----------
        val_loss_ : float
            The new validation loss.
        vf_var_ : tf.Variable
            The new id Visual Fields.
        g_var_ : tf.Variable
            The new generic Visual Field.
        iteration_ : int
            The iteration at which the new validation loss occurred.
        """
        if val_loss_ < self.val_loss:
            self.val_loss = val_loss_
            self.iteration = iteration_
            self.vf = vf_var_.numpy()
            self.g = g_var_.numpy()


class TrainLossesRecord:
    """Record the training and validation losses.

    Attributes
    ----------
    losses : list[float, ...]
        The training losses.
    losses_val : list[float, ...]
        The validation losses.

    Methods
    -------
    record(loss_, val_loss_)
        Add the new losses to the records.
    """
    def __init__(self):
        self.losses = []
        self.losses_val = []

    def record(self, loss_, val_loss_):
        """Add the new losses to the records.

        Parameters
        ----------
        loss_ : float
            The new training loss.
        val_loss_ : float
            The new validation loss.
        """
        self.losses.append(loss_)
        self.losses_val.append(val_loss_)


def optimize():
    """Optimize the Visual Fields (VFs).

    Optimizes the VFs till the validation loss stops decreasing.
    The training data is obtained from the rendered VF boundaries for
    ``NUMBERS.num_rand`` random faces and validation data from
    ``NUMBERS.num_val`` random faces.

    Returns
    -------
    LowestValLoss
        The Visual Fields corresponding to the lowest validation loss.
    """
    vf_var = tf.Variable(VF)
    g_var = tf.Variable(G)
    train_losses_record = TrainLossesRecord()
    lowest_val_loss = LowestValLoss()
    with tqdm() as pbar:
        iteration = 0
        while iteration - lowest_val_loss.iteration < NUMBERS.patience:
            if (iteration > NUMBERS.training_patience and
                    monotonically_increasing(train_losses_record, NUMBERS.training_patience)):
                break  # Training loss has been increasing for the last training_patience iterations
            if any_nan(vf_var).numpy() or any_nan(g_var).numpy():
                break  # NaN found
            loss_at_step, val_loss_at_step = gradient_step(vf_var, g_var, NUMBERS.learning_rate)
            train_losses_record.record(loss_at_step, val_loss_at_step)
            lowest_val_loss.record(val_loss_at_step, vf_var, g_var, iteration)
            iteration += 1
            pbar.set_description(f'Iteration {iteration}')
            _ = pbar.update(1)
    save_npy_files({
        DIRECTORIES.num_rand / NAMING.id.optimized.theta_boundary.npy: lowest_val_loss.vf,
        DIRECTORIES.num_rand / NAMING.generic.optimized.theta_boundary.npy: lowest_val_loss.g,
        DIRECTORIES.num_rand / NAMING.optimization.losses.npy: train_losses_record.losses,
        DIRECTORIES.num_rand / NAMING.optimization.losses.val.npy: train_losses_record.losses_val
    })
    return lowest_val_loss


def create_predictions(visual_fields, generic):
    """Create predictions for the Visual Fields of the random faces.

    Create predictions for the Visual Fields (VFs) of random faces
    from the optimized id and generic VFs.
    Saves the predictions in the :py:obj:`NAMING.optimization.predictions.npy`
    file.

    Parameters
    ----------
    visual_fields : np.ndarray
        The Visual Fields corresponding to the id parameters.
    generic : np.ndarray
        The Visual Field corresponding to the generic mesh.

    Returns
    -------
    None
    """
    predictions = predict(visual_fields, generic).numpy()
    predictions_val = predict(visual_fields, generic, params_=PARAMS_VAL).numpy()
    save_npy_files({
        DIRECTORIES.num_rand / NAMING.optimization.predictions.npy:
            np.concatenate([predictions_val, predictions], axis=0)
    })


def main():
    """Optimize the Visual Fields and create predictions.

    Optimize the Visual Field boundaries of id+, id- and generic faces and
    create predictions for the random faces using the linear model.
    """
    tf.random.set_seed(NUMBERS.tf_seed)
    lowest_val_loss = optimize()
    create_predictions(lowest_val_loss.vf, lowest_val_loss.g)


if __name__ == '__main__':
    main()
