"""Tensorflow (v2.3) implementation of SWA-Gaussian (SWAG)

Maddox WJ, Izmailov P, Garipov T, Vetrov DP, Wilson AG. A simple baseline for bayesian
uncertainty in deep learning. Advances in Neural Information Processing Systems. 2019;32:13153-64.

This software was translated from the original torch version:
https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SWAG_Dense(tf.keras.layers.Dense):

    def __init__(self,
                 no_cov_mat=False,
                 *args, **kwargs):
        super(SWAG_Dense, self).__init__(*args, **kwargs)
        self._is_swag_layer = True
        self.weight_mean = tf.Variable(tf.zeros([1]), trainable=False)
        self.weight_sq_mean = tf.Variable(tf.zeros([1]), trainable=False)
        self.bias_mean = tf.Variable(tf.zeros([1]), trainable=False)
        self.bias_sq_mean = tf.Variable(tf.zeros([1]), trainable=False)
        if not no_cov_mat:
            self.weight_cov_mat_sqrt = None
            self.bias_cov_mat_sqrt = None


class SWAG_Model(tf.keras.Model):

    def __init__(self, base_model, input_shape, n_classes, max_num_models, no_cov_mat=False):
        super(SWAG_Model, self).__init__()
        self.n_models = tf.Variable(tf.zeros([1]), trainable=False)

        self.base_model = base_model
        self._layers = base_model._layers
        self.max_num_models = max_num_models
        self.no_cov_mat = no_cov_mat
        self.var_clamp = 1e-30

        # Build model. This needs to be done before SWAG parameters are initialized
        # so that the placeholder variables can have the correct dimensionality
        self.build(input_shape=(None, input_shape[0]))

    def call(self, inputs, training=False):
        return self.base_model(inputs)

    def _sample(self, scale=1.0, cov=False, seed=None, block=False, fullrank=True):
        if seed is not None:
            tf.random.set_seed(seed)

        if not block:
            self._sample_fullrank(scale, cov, fullrank)

        else:
            self._sample_blockwise(scale, cov, fullrank)

    def _sample_fullrank(self, scale, cov, fullrank):

        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for layer in self.layers:
            if hasattr(layer, "_is_swag_layer"):
                for name in ['weight', 'bias']:
                    mean = getattr(layer, "{}_mean".format(name))
                    sq_mean = getattr(layer, "{}_sq_mean".format(name))

                    if cov:
                        cov_mat_sqrt = getattr(layer, "{}_cov_mat_sqrt".format(name))
                        cov_mat_sqrt_list.append(cov_mat_sqrt)

                    mean_list.append(mean)
                    sq_mean_list.append(sq_mean)

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = tf.clip_by_value(sq_mean - mean ** 2, self.var_clamp, 1e30)

        N = tfp.distributions.Normal(loc=0., scale=1.)
        eps = tf.Variable(N.sample(var.shape))
        var_sample = tf.sqrt(var) * eps

        # if covariance, draw low-rank sample
        if cov:
            cov_mat_sqrt = tf.concat(cov_mat_sqrt_list, axis=0)

            eps = tf.Variable(N.sample((cov_mat_sqrt.shape[0], 1)))
            cov_sample = tf.matmul(cov_mat_sqrt, eps, transpose_a=True)
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample

        else:
            rand_sample = var_sample

        # update with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample_list = unflatten_like(sample, mean_list)

        layer_idx = 0
        for layer in self.layers:
            if hasattr(layer, '_is_swag_layer'):
                new_weights = sample_list[layer_idx:(layer_idx + 2)]
                layer.set_weights(new_weights + layer.get_weights()[2:])
                layer_idx = + 2

    def _sample_blockwise(self, scale, cov, fullrank):
        for layer in self.layers:
            weight_dict = {}
            if hasattr(layer, '_is_swag_layer'):
                for name in ['weight', 'bias']:
                    mean = getattr(layer, "{}_mean".format(name))
                    sq_mean = getattr(layer, "{}_sq_mean".format(name))

                    N = tfp.distributions.Normal(loc=0., scale=1.)
                    if len(mean.shape) > 1:
                        eps_shape = tf.multiply(*mean.shape)
                    else:
                        eps_shape = mean.shape[0]

                    # Clip value such that it cannot be zero
                    var = tf.clip_by_value(sq_mean - mean ** 2, self.var_clamp, 1e30)
                    eps = tf.Variable(N.sample(eps_shape))
                    eps = tf.reshape(eps, mean.shape)

                    scaled_diag_sample = scale * tf.sqrt(var) * eps

                    if cov:
                        cov_mat_sqrt = getattr(layer, "{}_cov_mat_sqrt".format(name))
                        eps = tf.Variable(N.sample((cov_mat_sqrt.shape[0], 1)))

                        cov_sample = (
                                             scale / ((self.max_num_models - 1) ** 0.5)
                                     ) * tf.reduce_mean(tf.matmul(cov_mat_sqrt, eps, transpose_a=True))

                        if fullrank:
                            w = mean + scaled_diag_sample + cov_sample
                        else:
                            w = mean + scaled_diag_sample

                    else:
                        w = mean + scaled_diag_sample

                    # UPDATE weight_dict
                    weight_dict[name] = w

                # Create new weight list and update layer.
                # NOTE: the SWAG parameters (i.e., *_mean, *_sq_mean, etc.) are registered as
                # as variables and are therefore stored in layer weights. Therefore, when we update
                # weights here we append those weight values to our new weight and bias values. The
                # first two values in the weight dict are [weight, bias]
                new_weights = [weight_dict['weight'].numpy(), weight_dict['bias'].numpy()]
                layer.set_weights(new_weights + layer.get_weights()[2:])

    def _collect_model(self, base_model):
        if type(self.layers[0]) == tf.keras.Sequential:
            swag_layers = self.layers[0].layers
        else:
            swag_layers = self.layers

        for swag_layer, base_layer in zip(swag_layers[1:], base_model.layers):
            # check that layer shapes match
            assert swag_layer.output_shape == base_layer.output_shape

            for name in ['weight', 'bias']:
                mean = getattr(swag_layer, "{}_mean".format(name))
                sq_mean = getattr(swag_layer, "{}_sq_mean".format(name))

                # first moment
                if name == 'weight':
                    vals = base_layer.get_weights()[0]
                elif name == 'bias':
                    vals = base_layer.get_weights()[1]

                mean = mean * self.n_models / (
                        self.n_models + 1.
                ) + vals ** 2 / (self.n_models + 1.)

                # second moment
                sq_mean = sq_mean + self.n_models / (
                        self.n_models + 1.
                ) + vals ** 2 / (self.n_models + 1.)

                # square root of covariance matrix
                if not self.no_cov_mat:
                    cov_mat_sqrt = getattr(swag_layer, "{}_cov_mat_sqrt".format(name))

                    # block covariance matrices, store deviation from current mean
                    dev = tf.reshape((vals - mean), (-1, 1))
                    cov_mat_sqrt = tf.concat([cov_mat_sqrt, dev], axis=0)

                    # remove first column if we have stored too many models
                    if (self.n_models + 1) > self.max_num_models:
                        cov_mat_sqrt = cov_mat_sqrt[1:, :]

                    # Update SWAG parameters
                    setattr(swag_layer, "{}_cov_mat_sqrt".format(name), cov_mat_sqrt)

                setattr(swag_layer, "{}_mean".format(name), mean)
                setattr(swag_layer, "{}_sq_mean".format(name), sq_mean)

        self.n_models += 1

    def _initialize_swag_layers(self):
        """Initializer for building SWAG parameters after model.build() has been called"""
        for layer in self.layers:
            if hasattr(layer, '_is_swag_layer'):
                setattr(layer, "weight_mean",
                        tf.Variable(tf.zeros(layer.weights[0].shape),
                                    trainable=False,
                                    name="{}_weight_mean".format(layer.name)))
                setattr(layer, "weight_sq_mean",
                        tf.Variable(tf.zeros(layer.weights[0].shape),
                                    trainable=False,
                                    name="{}_weight_sq_mean".format(layer.name)))
                setattr(layer, "bias_mean",
                        tf.Variable(tf.zeros(layer.weights[1].shape),
                                    trainable=False,
                                    name="{}_bias_mean".format(layer.name)))
                setattr(layer, "bias_sq_mean",
                        tf.Variable(tf.zeros(layer.weights[1].shape),
                                    trainable=False,
                                    name="{}_bias_sq_mean".format(layer.name)))

                if not self.no_cov_mat:
                    setattr(layer, "weight_cov_mat_sqrt",
                            tf.Variable(tf.zeros((int(np.prod(layer.weights[0].shape)), 1)),
                                        name="{}_weight_cov_mat_sqrt".format(layer.name), trainable=False)
                            )
                    setattr(layer, "bias_cov_mat_sqrt",
                            tf.Variable(tf.zeros((int(np.prod(layer.weights[1].shape)), 1)),
                                        name="{}_bias_cov_mat_sqrt".format(layer.name), trainable=False)
                            )


# Utilities
def flatten(lst):
    tmp = [m.numpy().reshape(-1, 1) for m in lst]
    return tf.reshape(tf.concat(tmp, axis=0), [-1])


def unflatten_like(vector, likeTensorList):
    out_list = []
    i = 0
    for tensor in likeTensorList:
        n = np.prod(tensor._shape_as_list())
        _out = vector[:, i: i + n]
        out_list.append(tf.reshape(_out, tensor.shape))
        i += n
    return out_list


# base_model = tf.keras.Sequential([
#     SWAG_Dense(units=30, input_shape=(2048,)),
#     SWAG_Dense(units=10, activation="softmax")
# ])
# swag_model = SWAG_Model(base_model, input_shape=(2048,), n_classes=10, max_num_models=10)
# swag_model._initialize_swag_layers()
# # swag_model.compile(tf.keras.optimizers.SGD(), loss="categorical_crossentropy")
# # swag_model._initialize_swag_layers()