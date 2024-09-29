import numpy as np
import tensorflow as tf

from common_params import DIRECTORIES, NAMING, NUMBERS, save_npy_files
from tf_funcs import predict, loss, loss_val, any_nan, VF, G, PARAMS_VAL


def gradient_step(vf_var_, g_var_, lr_):
    with tf.GradientTape() as tape:
        loss_temp_ = loss(vf_var_, g_var_)
        dl_dvf_, dl_dg_ = tape.gradient(loss_temp_, [vf_var_, g_var_])
        _ = vf_var_.assign_sub(lr_ * dl_dvf_)
        _ = g_var_.assign_sub(lr_ * dl_dg_)
    loss_val_temp_ = loss_val(vf_var_, g_var_)
    return loss_temp_.numpy(), loss_val_temp_.numpy()


def monotonically_increasing(losses_record_, n_):
    return np.all(np.diff(losses_record_.losses[-n_:]) >= 0)


class LowestValLoss:
    def __init__(self):
        self.val_loss = float('inf')
        self.iteration = 0
        self.vf = None
        self.g = None

    def record(self, val_loss_, vf_var_, g_var_, iteration_):
        if val_loss_ < self.val_loss:
            self.val_loss = val_loss_
            self.iteration = iteration_
            self.vf = vf_var_.numpy()
            self.g = g_var_.numpy()


class TrainLossesRecord:
    def __init__(self):
        self.losses = []
        self.losses_val = []

    def record(self, loss_, val_loss_):
        self.losses.append(loss_)
        self.losses_val.append(val_loss_)


def optimize():
    vf_var = tf.Variable(VF)
    g_var = tf.Variable(G)
    train_losses_record = TrainLossesRecord()
    lowest_val_loss = LowestValLoss()
    iteration = 0
    while iteration - lowest_val_loss.iteration < NUMBERS.patience:
        if iteration % 100 == 0:
            print(f'Iteration: {iteration}')
        if iteration > NUMBERS.training_patience and monotonically_increasing(train_losses_record,
                                                                              NUMBERS.training_patience):
            break  # Training loss has been increasing for the last training_patience iterations
        if any_nan(vf_var).numpy() or any_nan(g_var).numpy():
            break  # NaN found
        loss_at_step, val_loss_at_step = gradient_step(vf_var, g_var, NUMBERS.learn_rate)
        train_losses_record.record(loss_at_step, val_loss_at_step)
        lowest_val_loss.record(val_loss_at_step, vf_var, g_var, iteration)
        iteration += 1
    save_npy_files({
        DIRECTORIES.vf / NAMING.npy.lowest_val_vf: lowest_val_loss.vf,
        DIRECTORIES.vf / NAMING.npy.lowest_val_g: lowest_val_loss.g,
        DIRECTORIES.vf / NAMING.npy.losses: train_losses_record.losses,
        DIRECTORIES.vf / NAMING.npy.losses_val: train_losses_record.losses_val
    })
    return lowest_val_loss


def create_predictions(visual_fields, generic):
    predictions = predict(visual_fields, generic).numpy()
    predictions_val = predict(visual_fields, generic, params_=PARAMS_VAL).numpy()
    save_npy_files({
        DIRECTORIES.vf / NAMING.npy.predictions: np.concatenate([predictions, predictions_val], axis=0)
    })


def main():
    tf.random.set_seed(NUMBERS.tf_seed)
    lowest_val_loss = optimize()
    create_predictions(lowest_val_loss.vf, lowest_val_loss.g)


if __name__ == '__main__':
    main()
