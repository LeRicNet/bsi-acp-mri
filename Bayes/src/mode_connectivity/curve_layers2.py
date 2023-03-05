"""TensorFlow Layers for Curve Models
"""
import numpy as np
import tensorflow as tf

class CurveModule(tf.Module):

    def __init__(self, fix_points, parameter_names, name=None):
        super(CurveModule, self).__init__(name=name)
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names,
        self.l2 = 0.0

    @tf.Module.with_name_scope
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("CurveModule.__call__ not implemented")

    @tf.Module.with_name_scope
    def compute_weights_t(self, coeffs_t):
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for coeff in coeffs_t.numpy():
                parameter = getattr(self, '%s_%s' % (parameter_name, self.name))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += tf.reduce_sum(w_t[i] ** 2)
        return w_t

class Dense(CurveModule):

    def __init__(
            self,
            in_features,
            out_features,
            fix_points,
            num_bends,
            activation=None,
            bias=True,
            parameter_names=('weight', 'bias'),
            name=None
    ):
        super(Dense, self).__init__(
            fix_points=fix_points,
            parameter_names=parameter_names,
            name=name
        )
        self.in_features = in_features
        self.out_features = out_features
        self.num_bends = num_bends
        self.activation = activation
        self.bias = bias

        with self.name_scope:
            for i, fixed in enumerate(self.fix_points):
                setattr(self, 'weight_%s' % self.name,
                        tf.Variable([in_features, out_features],
                                    trainable=not fixed, 
                                    name="curveDense_weight_{}".format(self.name)))

            for i, fixed in enumerate(self.fix_points):
                if bias:
                    setattr(self, 'bias_%s' % self.name,
                            tf.Variable([out_features], 
                                        trainable=not fixed, 
                                        name="curveDense_bias_{}".format(self.name)))
                else:
                    setattr(self, 'bias_%s' % self.name, None)
        
        self.reset_parameters()
        
    @tf.Module.with_name_scope
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.in_features)
        for i in range(self.num_bends):
            setattr(self, 'weight_%s' % self.name,
                    tf.Variable(lambda: tf.random.uniform(shape=(self.in_features, self.out_features),
                                                          minval=-stdv, maxval=stdv), 
                                name="curveDense_weight_{}".format(self.name)))

            bias = getattr(self, 'bias_%s' % self.name)
            if bias is not None:
                setattr(self, 'bias_%s' % self.name,
                        tf.Variable(lambda: tf.random.uniform(shape=[self.out_features],
                                                              minval=-stdv, maxval=stdv),
                                    name="curveDense_bias_{}".format(self.name)))
    
    @tf.Module.with_name_scope
    def __call__(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        output = tf.matmul(input, weight_t, transpose_b=False)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
        


