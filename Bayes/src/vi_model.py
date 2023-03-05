"""Variational Inference Posterior

https://github.com/wjmaddox/drbayes/blob/ccb11ca10530beb3e897627d94f478c4e20c4280/subspace_inference/posteriors/vi_model.py
"""

import tensorflow as tf

class VIModel(tf.keras.layers.Layer):

    def __init__(self, base_model, subspace, init_inv_softplus_sigma=-3.,
                 prior_log_sigma=3., eps=1e-6, with_mu=True):
        super(VIModel, self).__init__()

        self.base_model = base_model
        self.base_params = None

        self.subspace = subspace
        self.rank = self.subspace.rank

        self.prior_log_sigma = prior_log_sigma
        self.eps = eps

        self.with_mu = with_mu
        if with_mu:
            self.mu = None
        self.inv_softplus_sigma = None

    def __call__(self, *args, **kwargs):
        sigma = tf.nn.softplus(self.inv_softplus_sigma) * self.eps
        if self.with_mu:
            z = self.mu
        else:
            z = None
        w = self.subspace(z)

        return self.base_model(*args, **kwargs)