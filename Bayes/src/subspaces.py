"""
    subspace classes
    CovarianceSpace: covariance subspace
    PCASpace: PCA subspace
    FreqDirSpace: Frequent Directions Space

Translated to TensorFlow from pyTorch:
https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/posteriors/subspaces.py
"""

import abc
import numpy as np
import tensorflow as tf

from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition.pca import _assess_dimension_
from sklearn.utils.extmath import randomized_svd
# import tensorflow_probability as tfp

from scipy.special import gammaln


# class Subspace(tf.keras.layers.Layer, metaclass=abc.ABCMeta):
class Subspace(tf.keras.Model, metaclass=abc.ABCMeta):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subspace_type):
        def decorator(subclass):
            cls.subclasses[subspace_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, subspace_type, **kwargs):
        if subspace_type not in cls.subclasses:
            raise ValueError('Bad subspaces type {}'.format(subspace_type))
        return cls.subclasses[subspace_type](**kwargs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init__(self):
        super(Subspace, self).__init__()

    @abc.abstractmethod
    def collect_vector(self, vector):
        pass

    @abc.abstractmethod
    def get_space(self):
        pass



@Subspace.register_subclass('random')
class RandomSpace(Subspace):
    def __init__(self, num_parameters, rank=20, method='dense'):
        assert method in ['dense', 'fastfood']

        super(RandomSpace, self).__init__()

        self.num_parameters = num_parameters
        self.rank = rank
        self.method = method

        if method == 'dense':
            # N = tfp.distributions.Normal(loc=0., scale=1.)
            # self.subspace = N.sample((rank, num_parameters))
            self.subspace = tf.random.normal((rank, num_parameters))

        if method == 'fastfood':
            raise NotImplementedError("FastFood transform hasn't been implemented yet")

    # random subspace is independent of data
    def collect_vector(self, vector):
        pass

    def get_space(self):
        return self.subspace


@Subspace.register_subclass('covariance')
class CovarianceSpace(Subspace):

    def __init__(self, num_parameters, max_rank=20):
        super(CovarianceSpace, self).__init__()

        self.num_parameters = num_parameters

        # self.rank = tf.Variable(tf.zeros(1), name="CovarianceSubspaceRank")
        # self.cov_mat_sqrt = tf.Variable(tf.zeros([1, self.num_parameters]),
        #                                 name="CovarianceSubspaceCovMatSqrt"))
        self.rank = 0
        self.cov_mat_sqrt = np.zeros(shape=(1, self.num_parameters))
        # self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        # self.register_buffer('cov_mat_sqrt',
        #                      torch.empty(0, self.num_parameters, dtype=torch.float32))
        self.max_rank = max_rank

    # def collect_vector(self, vector):
    #     if self.rank + 1 > self.max_rank:
    #         self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
    #
    #     self.cov_mat_sqrt = tf.concat([self.cov_mat_sqrt, tf.reshape(vector, -1)], axis=1)
    #     print(self.cov_mat_sqrt)
    #     # self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
    #     self.rank = tf.minimum(self.rank + 1, tf.convert_to_tensor(self.max_rank).set_shape(-1))
    #     # self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)

    def collect_vector(self, vector):
        if self.rank + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]

        self.cov_mat_sqrt = np.concatenate([self.cov_mat_sqrt, vector.numpy().reshape(1,-1)], axis=0)
        self.rank = np.minimum((self.rank + 1), self.max_rank)

    def get_space(self):
        return np.copy(self.cov_mat_sqrt, deep=True) / (self.cov_mat_sqrt.shape[0] - 1) ** 0.5

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        rank = state_dict[prefix + 'rank'].item()
        self.cov_mat_sqrt = self.cov_mat_sqrt.new_empty((rank, self.cov_mat_sqrt.shape[1]))
        super(CovarianceSpace, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                           strict, missing_keys, unexpected_keys,
                                                           error_msgs)


@Subspace.register_subclass('pca')
class PCASpace(CovarianceSpace):

    def __init__(self, num_parameters, pca_rank='mle', max_rank=100):
        super(PCASpace, self).__init__(num_parameters, max_rank=max_rank)

        # better phrasing for this condition?
        assert (pca_rank == 'mle' or isinstance(pca_rank, int))
        if pca_rank != 'mle':
            assert 1 <= pca_rank <= max_rank

        self.pca_rank = pca_rank

    def get_config(self):
        return {
            'pca_rank': self.pca_rank,
            'num_parameters': self.num_parameters,
            'max_rank': self.max_rank
        }

    def get_space(self):

        cov_mat_sqrt_np = tf.raw_ops.Copy(input=self.cov_mat_sqrt).numpy()

        # perform PCA on DD'
        cov_mat_sqrt_np /= (max(1, self.rank.numpy() - 1)) ** 0.5

        if self.pca_rank == 'mle':
            pca_rank = self.rank.numpy()
        else:
            pca_rank = self.pca_rank

        pca_rank = max(1, min(pca_rank, self.rank.numpy()))
        pca_decomp = TruncatedSVD(n_components=pca_rank)
        pca_decomp.fit(cov_mat_sqrt_np)

        _, s, Vt = randomized_svd(cov_mat_sqrt_np, n_components=pca_rank, n_iter=5)

        # perform post-selection fitting
        if self.pca_rank == 'mle':
            eigs = s ** 2.0
            ll = np.zeros(len(eigs))
            correction = np.zeros(len(eigs))

            # compute minka's PCA marginal log likelihood and the correction term
            for rank in range(len(eigs)):
                # secondary correction term based on the rank of the matrix + degrees of freedom
                m = cov_mat_sqrt_np.shape[1] * rank - rank * (rank + 1) / 2.
                correction[rank] = 0.5 * m * np.log(cov_mat_sqrt_np.shape[0])
                ll[rank] = _assess_dimension_(spectrum=eigs,
                                              rank=rank,
                                              n_features=min(cov_mat_sqrt_np.shape),
                                              n_samples=max(cov_mat_sqrt_np.shape))

            self.ll = ll
            self.corrected_ll = ll - correction
            self.pca_rank = np.nanargmax(self.corrected_ll)
            print('PCA Rank is: ', self.pca_rank)
            return tf.Tensor(s[:self.pca_rank, None] * Vt[:self.pca_rank, :])
        else:
            return tf.Tensor(s[:, None] * Vt)


def _assess_dimension_(spectrum, rank, n_samples, n_features):
    """Compute the likelihood of a rank ``rank`` dataset

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``.

    Parameters
    ----------
    spectrum : array of shape (n)
        Data spectrum.
    rank : int
        Tested rank value.
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.

    Returns
    -------
    ll : float,
        The log-likelihood

    Notes
    -----
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    """
    if rank > len(spectrum):
        raise ValueError("The tested rank cannot exceed the rank of the"
                         " dataset")

    pu = -rank * log(2.)
    for i in range(rank):
        pu += (gammaln((n_features - i) / 2.) -
               log(np.pi) * (n_features - i) / 2.)

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.

    if rank == n_features:
        pv = 0
        v = 1
    else:
        v = np.sum(spectrum[rank:]) / (n_features - rank)
        pv = -np.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = log(2. * np.pi) * (m + rank + 1.) / 2.

    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2. - rank * log(n_samples) / 2.

    return ll
