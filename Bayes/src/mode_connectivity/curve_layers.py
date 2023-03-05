"""TensorFlow Layers for Curve Models
"""
import numpy as np
import tensorflow as tf

class CurveModule(tf.keras.layers.Layer):
    """Base class for building a Curve network"""

    def __init__(self, fix_points, parameter_names=(), **kwargs):
        super(CurveModule, self).__init__(dynamic=True, **kwargs)
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        # self.l2 = 0.0

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("CurveModule.__call__ not implemented")

    # @tf.function
    def compute_weights_t(self, coeffs_t):
        """Collect the N replicate weights/biases where N=1:num_bends"""

        w_t = []
        for i, parameter in enumerate(self.parameter_names):
            # _w_t = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            _w_t = []
            params = [p for p in self.weights if parameter in p.name]
            for j in range(coeffs_t.shape[0]):
                if params[j].trainable:
                    _w_t.append(params[j] * coeffs_t[j])
                else:
                    _w_t.append(params[j])

            # w_t = w_t.write(i, _w_t.stack())

            w_t.append(tf.squeeze(tf.reduce_sum(_w_t, axis=0)))

        # w_t = w_t.stack()
        for w in w_t:
            self.add_loss(tf.reduce_sum(w ** 2))
        return w_t

        # print(len(w_t))
        # if len(w_t) == 1:
        #     return tf.transpose(tf.expand_dims(w_t, 0))
        # else:
        #     return tf.transpose(w_t)
    # def compute_weights_t(self, coeffs_t):
    #     w_t = [None] * len(self.parameter_names)
    #     self.l2 = 0.0
    #     for i, parameter_name in enumerate(self.parameter_names):
    #         compute_weights = lambda j, coeff: self.__compute_weights_iterator_fn(parameter_name, [i], j, coeff)
    #         wt[i] = tf.map_fn(compute_weights,
    #                           elems=enumerate(coeffs_t))
    #         if w_t[i] is not None:
    #             self.l2 += tf.reduce_sum(w_t[i] ** 2)
    #     return w_t
    #
    # def __compute_weights_iterator_fn(self, parameter_name, w_t_i, j, coeff):
    #     parameter = getattr(self, '%s_%d' % (parameter_name, j))
    #     if parameter is not None:
    #         if w_t_i is None:
    #             w_t_i = parameter * coeff
    #         else:
    #             w_t_i += parameter * coeff
    #     return w_t_i

class Dense(CurveModule):

    def __init__(self, in_features, out_features, fix_points,
                 num_bends, activation=None, bias=True):
        super(Dense, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.bias = bias
        self.num_bends = num_bends

        self.l2 = 0.0

        stdv = 1. / np.sqrt(self.in_features)
        for i, fixed in enumerate(self.fix_points):
            self.add_weight(name='weight_%s_%i' % (self.name, i), trainable=not fixed,
                            initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv),
                            shape=(in_features, out_features), dtype=tf.float32)
            if self.bias:
                self.add_weight(name='bias_%s_%i' % (self.name, i), trainable=not fixed,
                                initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv),
                                shape=[out_features], dtype=tf.float32)
        #
        #
        #
        #         setattr(self, 'weight_%s_%i' % (self.name, i),
        #                 tf.Variable([in_features, out_features],
        #                             trainable=not fixed,
        #                             name='weight_%s_%i' % (self.name, i)))
        #         # tape.watch(getattr(self, 'weight_%s_%i' % (self.name, i)))
        #
        #     for i, fixed in enumerate(self.fix_points):
        #         if bias:
        #             setattr(self, 'bias_%s_%i' % (self.name, i),
        #                     tf.Variable([out_features], trainable=not fixed,
        #                                 name='bias_%s_%i' % (self.name, i)))
        #         else:
        #             setattr(self, 'bias_%s_%i' % (self.name, i), None)
        #         # tape.watch(getattr(self, 'bias_%s_%i' % (self.name, i)))
        #
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / np.sqrt(self.in_features)
    #     with tf.GradientTape() as tape:
    #         for i in range(self.num_bends):
    #             setattr(self, 'weight_%s_%i' % (self.name, i),
    #                     tf.Variable(lambda: tf.random.uniform(shape=(self.in_features, self.out_features),
    #                                                           minval=-stdv, maxval=stdv), name="curveDense_weight"))
    #
    #             bias = getattr(self, 'bias_%s_%i' % (self.name, i))
    #             if bias is not None:
    #                 setattr(self, 'bias_%s_%i' % (self.name, i),
    #                         tf.Variable(lambda: tf.random.uniform(shape=[self.out_features],
    #                                                               minval=-stdv, maxval=stdv), name="curveDense_bias"))
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.in_features)
        with tf.GradientTape(persistent=True) as tape:
            for i in range(self.num_bends):
                setattr(self, 'weight_%s_%i' % (self.name, i),
                        tf.random.uniform(shape=(self.in_features, self.out_features),
                                          minval=-stdv, maxval=stdv))

                bias = getattr(self, 'bias_%s_%i' % (self.name, i))
                if bias is not None:
                    setattr(self, 'bias_%s_%i' % (self.name, i),
                            tf.random.uniform(shape=[self.out_features],
                                              minval=-stdv, maxval=stdv))

    def __call__(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        output = tf.matmul(input, weight_t, transpose_b=False)
        if self.bias is not None:
            output = output + bias_t
        if self.activation is not None:
            output = self.activation(output)
        return output

# class Dense(CurveModule):
#
#     def __init__(self, in_features, out_features, fix_points,
#                  num_bends, activation=None, bias=True):
#         super(Dense, self).__init__(fix_points, ('weight', 'bias'))
#         self.in_features = in_features
#         self.out_features = out_features
#         self.activation = activation
#         self.bias = bias
#         self.num_bends = num_bends
#
#         self.l2 = 0.0
#         with tf.GradientTape() as tape:
#             for i, fixed in enumerate(self.fix_points):
#                 setattr(self, 'weight_%s_%i' % (self.name, i),
#                         tf.Variable([in_features, out_features],
#                                     trainable=not fixed,
#                                     name='weight_%s_%i' % (self.name, i)))
#                 # tape.watch(getattr(self, 'weight_%s_%i' % (self.name, i)))
#
#             for i, fixed in enumerate(self.fix_points):
#                 if bias:
#                     setattr(self, 'bias_%s_%i' % (self.name, i),
#                             tf.Variable([out_features], trainable=not fixed,
#                                         name='bias_%s_%i' % (self.name, i)))
#                 else:
#                     setattr(self, 'bias_%s_%i' % (self.name, i), None)
#                 # tape.watch(getattr(self, 'bias_%s_%i' % (self.name, i)))
#
#         self.reset_parameters()
#
#     # def reset_parameters(self):
#     #     stdv = 1. / np.sqrt(self.in_features)
#     #     with tf.GradientTape() as tape:
#     #         for i in range(self.num_bends):
#     #             setattr(self, 'weight_%s_%i' % (self.name, i),
#     #                     tf.Variable(lambda: tf.random.uniform(shape=(self.in_features, self.out_features),
#     #                                                           minval=-stdv, maxval=stdv), name="curveDense_weight"))
#     #
#     #             bias = getattr(self, 'bias_%s_%i' % (self.name, i))
#     #             if bias is not None:
#     #                 setattr(self, 'bias_%s_%i' % (self.name, i),
#     #                         tf.Variable(lambda: tf.random.uniform(shape=[self.out_features],
#     #                                                               minval=-stdv, maxval=stdv), name="curveDense_bias"))
#     def reset_parameters(self):
#         stdv = 1. / np.sqrt(self.in_features)
#         with tf.GradientTape(persistent=True) as tape:
#             for i in range(self.num_bends):
#                 setattr(self, 'weight_%s_%i' % (self.name, i),
#                         tf.random.uniform(shape=(self.in_features, self.out_features),
#                                           minval=-stdv, maxval=stdv))
#
#                 bias = getattr(self, 'bias_%s_%i' % (self.name, i))
#                 if bias is not None:
#                     setattr(self, 'bias_%s_%i' % (self.name, i),
#                             tf.random.uniform(shape=[self.out_features],
#                                               minval=-stdv, maxval=stdv))
#
#     def __call__(self, input, coeffs_t):
#         weight_t, bias_t = self.compute_weights_t(coeffs_t)
#         output = tf.matmul(input, weight_t, transpose_b=False)
#         if self.bias is not None:
#             output = output + self.bias
#         if self.activation is not None:
#             output = self.activation(output)
#         return output

class _BatchNorm(CurveModule):
    _version = 2

    def __init__(self, num_features, fix_points, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, training=True, **kwargs):
        super(_BatchNorm, self).__init__(fix_points, ('weight', 'bias'), **kwargs)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.training = training

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                setattr(self, 'weight_%s' % self.name,
                        tf.Variable(lambda: tf.random.uniform([num_features]), trainable=not fixed))
            else:
                setattr(self, 'weight_%s' % self.name, None)

        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                setattr(self, 'bias_%s' % self.name,
                        tf.Variable(lambda: tf.random.uniform([num_features]), trainable=not fixed))

            else:
                setattr(self, 'bias_%s' % self.name, None)

        if self.track_running_stats:
            with tf.GradientTape(persistent=True) as tape:
                setattr(self, 'running_mean_%s' % self.name,
                        tf.Variable(tf.zeros(num_features), trainable=True))
                setattr(self, 'running_var_%s' % self.name,
                        tf.Variable(tf.ones(num_features), trainable=True))

            # self.running_mean = tf.Variable(tf.zeros(num_features))
            # self.running_var = tf.Variable(tf.ones(num_features))
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            with tf.GradientTape(persistent=True) as tape:
                setattr(self, 'running_mean_%s' % self.name,
                        tf.Variable(lambda: tf.zeros(self.num_features), trainable=True))
                setattr(self, 'running_var_%s' % self.name,
                        tf.Variable(lambda: tf.ones(self.num_features), trainable=True))
            self.num_batches_tracked = 0

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for i, fixed in enumerate(self.fix_points):
                setattr(self, 'weight_%s' % self.name,
                        tf.Variable(lambda: tf.random.uniform([self.num_features]),
                                    trainable=not fixed))
                setattr(self, 'bias_%s' % self.name,
                        tf.Variable(lambda: tf.zeros([self.num_features]),
                                    trainable=not fixed))

    def _check_input_dim(self, input):
        raise NotImplementedError

    def __call__(self, input, coeffs_t, *args, **kwargs):
        self._check_input_dim(input)
        # print('_BatchNorm.__call__: {}'.format(coeffs_t))

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        if not self.training:
            return input
        else:
            # print('_BatchNorm.__call__: {}'.format(coeffs_t))
            weight_t, bias_t = self.compute_weights_t(coeffs_t)
            # bias_t = None
            running_mean = getattr(self, "running_mean_%s" % self.name)
            running_var = getattr(self, 'running_var_%s' % self.name)
            
            # DEBUGGING
            print("BN, weight_t: {}".format(weight_t))
            print("BN, bias_t: {}".format(bias_t))
            print("BN, running_mean: {}".format(running_mean))
            print("BN, running_var: {}".format(running_var))
            
            output = tf.nn.batch_normalization(input, running_mean, running_var, bias_t, weight_t, self.eps)
            print("BN, output: {}".format(output))
            return output

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    # def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     version = metadata.get('version', None)
    #
    #     if (version is None or version < 2) and self.track_running_stats:
    #         # at version 2: added num_batches_tracked buffer
    #         #               this should have a default value of 0
    #         num_batches_tracked_key = prefix + 'num_batches_tracked'
    #         if num_batches_tracked_key not in state_dict:
    #             state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
    #
    #     super(_BatchNorm, self)._load_from_state_dict(
    #         state_dict, prefix, metadata, strict,
    #         missing_keys, unexpected_keys, error_msgs)

class BatchNormalization(_BatchNorm):

    def _check_input_dim(self, input):
        if len(input.shape) != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(len(input.shape)))

class BatchNormalization2d(_BatchNorm):

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(len(input.shape)))

class Linear(CurveModule):

    def __init__(self, in_features, out_features, fix_points, activation=None, bias=True):
        super(Linear, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            setattr(self, 'weight_%d' % i, tf.Variable([out_features, in_features], trainable=not fixed))

        for i, fixed in enumerate(self.fix_points):
            if bias:
                setattr(self, 'bias_%d' % i, tf.Variable([out_features], trainable=not fixed))
            else:
                setattr(self, 'bias_%d' % i, None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.in_features)
        for i in range(self.num_bends):
            setattr(self, 'weight_%d' % i, tf.random.uniform(shape=(self.out_features, self.in_features),
                                                             minval=-stdv, maxval=stdv))

            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                setattr(self, 'bias_%d' % i, tf.random.uniform(shape=[self.out_features], minval=-stdv, maxval=stdv))

    def __call__(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        output = tf.matmul(input, tf.transpose(weight_t))
        if bias is not None:
            output += bias
        if self.activation is not None:
            output = self.activation(output)
        return output

class Conv2d(CurveModule):

    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride=1,
                 padding='VALID', dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(fix_points, ('weight', 'bias'))
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        for i, fixed in enumerate(self.fix_points):
            setattr(self, 'weight_%d' % i, tf.Variable([out_channels, in_channels // groups], shape=kernel_size,
                                                       trainable=not fixed))
        for i, fixed in enumerate(self.fix_points):
            if bias:
                setattr(self, 'bias_%d' % i, tf.Variable([out_channels], trainable=not fixed))
            else:
                setattr(self, 'bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / tf.sqrt(n)
        for i in range(self.num_bends):
            setattr(self, 'weight_%d' % i, tf.random.uniform(shape=(self.out_features, self.in_features),
                                                             minval=-stdv, maxval=stdv))
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                setattr(self, 'bias_%d' % i, tf.random.uniform(shape=(self.out_features),
                                                               minval=-stdv, maxval=stdv))

    def __call__(self, input, coeffs_t, *args, **kwargs):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        output = tf.nn.conv2d(input, weight_t, bias_t, self.stride, self.padding, self.dilation)  # Not using self.groups parameter.
        return output