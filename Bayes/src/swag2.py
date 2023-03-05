import numpy as np
import tensorflow as tf
from .subspaces import Subspace
from .train_utils import set_weights

class SWAG(tf.keras.Model):

    def __init__(self, base_model, subspace_type,
                 subspace_kwargs=None, var_clamp=1e-6, *args, **kwargs):
        super(SWAG, self).__init__()

        self.base_model = tf.keras.models.clone_model(base_model)
        self.num_parameters = sum(np.prod(l.get_weights()[0].shape) for l in self.base_model.layers if len(l.get_weights()) > 1)

        self.mean = tf.Variable(tf.zeros(self.num_parameters), trainable=False, name="swag_mean")
        self.sq_mean = tf.Variable(tf.zeros(self.num_parameters), trainable=False, name="swag_sq_mean")
        self.n_models = tf.Variable(tf.zeros(1, dtype=tf.float32), trainable=False, name="swag_n_models")

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = 'cpu'


    # def build(self):
    #     pass

    def call(self, inputs, **kwargs):
        return self.base_model(inputs, **kwargs)

    def _get_mean_and_variance(self):
        variance = tf.clip_by_value(self.sq_mean - self.mean ** 2, self.var_clamp, 1e6)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        w = flatten([l.get_weights()[0] for l in base_model.layers if len(l.get_weights()) > 1])
        # first moment
        mean = tf.multiply(self.mean, (self.n_models / (self.n_models + 1.)))
        self.mean = mean + (w / (self.n_models + 1.))

        # second moment
        sq_mean = tf.multiply(self.sq_mean, (self.n_models / (self.n_models + 1.)))
        self.sq_mean = sq_mean + (w ** 2 / (self.n_models + 1.))

        # deviations
        dev_vector = w - self.mean

        # update necessary objects
        self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models = self.n_models + 1

    def set_swa(self):
        set_weights(self.base_model, self.mean, device=None)

    # def save


def flatten(lst):
    tmp = [tf.reshape(x, (-1, 1)) for x in lst]
    tmp = tf.reshape(tf.concat(tmp, axis=0), -1)
    return tmp