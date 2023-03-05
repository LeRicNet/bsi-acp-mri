import tensorflow as tf

class SubspaceModel(tf.keras.layers.Layer):
    def __init__(self, mean, cov_factor):
        super(SubspaceModel, self).__init__()
        self.rank = cov_factor.shape[0]
        self.mean = mean
        self.cov_factor = cov_factor

    def call(self, t):
        return self.mean + tf.matmul(tf.transpose(self.cov_factor), t)


class ProjectedModel(tf.keras.layers.Layer):
    def __init__(self, proj_params, model, projection=None, mean=None, subspace=None):
        super(ProjectedModel, self).__init__()
        self.model = model

        if subspace is None:
            self.subspace = SubspaceModel(mean, projection)
        else:
            self.subspace = subspace

        if mean is None and subspace is None:
            raise NotImplementedError('Must enter either subspace or mean')

        self.proj_params = proj_params