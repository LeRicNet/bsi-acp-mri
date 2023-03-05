from collections import abc
import numpy as np
import tensorflow as tf

def _ntuple(n):
    def parse(x):
        if isinstance(x, abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2


# def learning_rate_schedule(base_lr, epoch, total_epochs):
#     alpha = epoch / total_epochs
#     if alpha <= 0.5:
#         factor = 1.0
#     elif alpha <= 0.9:
#         factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
#     else:
#         factor = 0.01
#     return factor * base_lr

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, total_epochs, batch_size, cardinality, **kwargs):
        super(LRSchedule, self).__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.total_epochs = total_epochs
        self.steps_per_epoch = int(np.ceil(cardinality / batch_size))
        self.epoch = 0

    def __call__(self, step):
        if step > 0 and step % self.steps_per_epoch == 0:
            self.epoch += 1
            alpha = self.epoch / self.total_epochs
            if alpha < 0.5:
                factor = 1.0
            elif alpha <= 0.9:
                factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
            else:
                factor = 0.01
            return factor * self.initial_learning_rate
        else:
            return self. initial_learning_rate
