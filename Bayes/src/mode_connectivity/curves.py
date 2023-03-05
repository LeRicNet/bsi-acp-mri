"""Mode Connectivity

TensorFlow r2.3 translation of the PyTorch r0.3.1 implementation.
https://github.com/timgaripov/dnn-mode-connectivity/blob/master/curves.py

@inproceedings{garipov2018loss,
  title={Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs},
  author={Garipov, Timur and Izmailov, Pavel and Podoprikhin, Dmitrii and Vetrov, Dmitry P and Wilson, Andrew Gordon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
"""
import numpy as np
from scipy.special import binom
import tensorflow as tf

from Bayes.src.mode_connectivity.utils import _pair, l2_regularizer
from Bayes.src.mode_connectivity.curve_layers import Dense, CurveModule, BatchNormalization
# from Bayes.src.mode_connectivity.curve_layers import CurveModule, BatchNormalization
# from Bayes.src.mode_connectivity.curve_layers2 import Dense


class Bezier(tf.keras.Model):
    # TODO: Define Bezier
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.binom = tf.Variable(binom(num_bends - 1, np.arange(num_bends)),
                                 trainable=False, dtype=tf.float32, name="bezier_binom")
        self.range = tf.range(0, num_bends, dtype=tf.float32)
        self.rev_range = tf.range((num_bends - 1), -1, -1, dtype=tf.float32)

    def __call__(self, t, *args, **kwargs):
        curve = self.binom * t ** self.range * (1.0 - t) ** self.rev_range
        return curve


class PolyChain(tf.keras.Model):
    # TODO: Define PolyChain
    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.range = tf.range(0, float(num_bends), dtype=tf.float32)

    def __call__(self, t):
        t_n = t * (self.num_bends - 1)
        return tf.math.maximum(0.0, 1.0 - tf.abs(t_n - self.range))


class CurveNet(tf.keras.Model):
    def __init__(self, num_classes, curve, architecture, num_bends, fix_start=True, fix_end=True,
                 architecture_kwargs={}):
        super(CurveNet, self).__init__()
        self.num_classes = num_classes
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]

        self.curve = curve
        self.architecture = architecture

        self.l2 = 0.0
        self.coeff_layer = self.curve(self.num_bends)
        # self.net = self.architecture(num_classes, fix_points=self.fix_points, **architecture_kwargs)
        self.net = self.architecture
        self.curve_modules = []
        for layer in self.net.layers:
            if issubclass(layer.__class__, CurveModule):
                self.curve_modules.append(layer)

    def import_base_parameters(self, base_model, index):

        weights = [v for v in base_model.variables if 'kernel' in v.name]
        biases = [v for v in base_model.variables if 'bias' in v.name]

        for curve_module, weight, bias in zip(self.curve_modules, weights, biases):
            _weights = [w for w in curve_module.weights if 'weight' in w.name]
            _biases = [b for b in curve_module.weights if 'bias' in b.name]
            replace_idx = [i for i in list(range(len(_weights)))[index::self.num_bends]]

            for idx in replace_idx:
                _weights[idx].assign(weight)
                _biases[idx].assign(bias)

            new_weights_and_biases = []
            for w, b in zip(_weights, _biases):
                new_weights_and_biases.extend([w.value(), b.value()])

            curve_module.set_weights(np.array(new_weights_and_biases))




        # for layer, base_layer in zip(self.net.layers[index::self.num_bends],
        #                              base_model.layers[index::self.num_bends]):
        #                              # base_model.layers):
        #     base_weights = base_layer.get_weights()
        #
        #     for i, fixed in enumerate(self.fix_points):
        #         if hasattr(layer, 'weight_%s_%i' % (layer.name, i)):
        #             setattr(layer, 'weight_%s_%i' % (layer.name, i), base_weights[0])
        #         if hasattr(layer, 'bias_%s_%i' % (layer.name, i)):
        #             setattr(layer, 'bias_%s_%i' % (layer.name, i), base_weights[1])


            # if layer in self.curve_modules:
            #     base_weights = base_weights * self.num_bends

            # if len(layer_weights) != len(base_weights):
            #     for bw in base_weights:
            #         layer.add_weight(shape=bw.shape)
                # layer.add_weight(shape=np.array(base_weights).shape)

    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(dir(self.net), dir(base_model)):
            setattr(self.net, buffer, getattr(base_model, base_buffer))

    def export_base_parameters(self, base_model, index):
        for layer, base_layer in zip(self.net.layers[index::self.num_bends],
                                     base_model.layers[index::self.num_bends]):
            base_layer.set_weights(layer.get_weights())

    def init_linear(self):

        for curve_layer in self.curve_modules:
            layer_weights = curve_layer.weights

            weights = [w for w in layer_weights if 'weight' in w.name]
            bias = [b for b in layer_weights if 'bias' in b.name]

            # The start (0) and end points (num_bends) are fixed points
            new_weights_and_biases = [weights[0].value(), bias[0].value()]

            for i in range(1, self.num_bends - 1):
                alpha = i * 1.0 / (self.num_bends - 1)
                new_weights = alpha * weights[-1] + (1.0 - alpha) * weights[0]
                new_bias = alpha * bias[-1] + (1.0 - alpha) * bias[0]
                new_weights_and_biases.extend([new_weights, new_bias])

            new_weights_and_biases.extend([weights[-1].value(), bias[-1].value()])
            curve_layer.set_weights(np.array(new_weights_and_biases))
            

    def _weights(self, t):
        coeffs_t = self.coeff_layer(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w.numpy() for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.ravel() for w in weights])

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def call(self, input, t=None, initializer=tf.random_uniform_initializer(), *args, **kwargs):
        if t is None:
            t = tf.Variable(initializer(shape=[1]), trainable=False, name="t")
            # t = input.data.new(1).uniform_()
        coeffs_t = tf.Variable(self.coeff_layer(t), trainable=True, name="coeffs_t")
        # self.weights = self._weights(t)
        output = self.net(input, coeffs_t, *args, **kwargs)
        self._compute_l2()
        return output, self.l2, self._weights(t)
