import tensorflow as tf

# Local Imports
from Bayes.src.mode_connectivity import curves

class BaseModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()

    def build(self, input_shape):
        self.bn_1 = tf.keras.layers.BatchNormalization(input_shape=(2048,))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.linear = tf.keras.layers.Dense(
                units=num_classes,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001, seed=SEED),
                bias_regularizer=tf.keras.regularizers.L2(),
                activation="linear")
        self.softplus = tf.keras.layers.Activation("softplus")
        self.softmax = tf.keras.layers.Softmax()


    def __call__(self, inputs, *args, **kwargs):
        net = self.bn_1(inputs)
        if training:
            net = self.dropout(net)
        net = self.linear(net)
        net = self.softplus(net)
        return self.softmax(net)

class BaseCurve(tf.keras.Model):
    def __init__(self, num_classes, fix_points):
        super(BaseCurve, self).__init__()
        self.num_classes = num_classes
        self.fix_points = fix_points
        self.base_model = None
        self.build(input_shape=None)

    def build(self, input_shape):

        # self.input_layer = tf.keras.layers.InputLayer(input_shape=2048)
        self.bn_1 = curves.BatchNorm2d(num_features=2048, fix_points=self.fix_points)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.linear = curves.Linear(in_features=2048, out_features=self.num_classes, fix_points=self.fix_points)
        self.softplus = tf.keras.layers.Activation("softplus")
        self.softmax = tf.keras.layers.Softmax()


    def __call__(self, inputs, coeffs_t=None, *args, **kwargs):
        # net = self.input_layer(inputs)
        net = self.bn_1(inputs, coeffs_t)
        if training:
            net = self.dropout(net)
        net = self.linear(net, coeffs_t)
        net = self.softplus(net)
        return self.softmax(net)
        # return self.base_model(inputs, *args, **kwargs)


class BaseNet:
    base = BaseModel
    curve = BaseCurve

def generate_curve_model(base_model, fix_start, fix_end, num_bends, num_classes=2):
    """Simple Curve Model Generation

    Given a tf.keras.Model, implement a CurveModule-extending class with the
    appropriate layers switched.

    Currently Implemented Layers:
        tf.keras.layers.BatchNormalization --> curves.BatchNormalization
        tf.keras.layers.Dense --> curves.Dense
    """

    IMPLEMENTED_LAYERS = [
        # tf.keras.layers.BatchNormalization,
        tf.keras.layers.Dense
    ]

    CURVE_LAYERS = [
        # curves.BatchNormalization,
        curves.Dense
    ]

    fix_points = [fix_start] + [False] * (num_bends - 2) + [fix_end]
    
    model_layers = base_model.layers
    curve_model_layers = []
    for layer in model_layers:
        if layer.__class__ in IMPLEMENTED_LAYERS:
            in_features = layer.input_shape[-1]
            out_features = layer.output_shape[-1]

            # if layer.__class__ == IMPLEMENTED_LAYERS[0]:
            #     assert in_features == out_features, "in_features != out_features"
            #     next_layer = curves.BatchNormalization(num_features=in_features,
            #                                            fix_points=fix_points)
            if layer.__class__ == IMPLEMENTED_LAYERS[0]:
                activation = layer.get_config()['activation']
                if activation is not None:
                    activation = tf.keras.activations.get(activation)
                next_layer = curves.Dense(in_features=in_features, out_features=out_features,
                                          fix_points=fix_points, num_bends=num_bends, activation=activation)
        else:
            next_layer = layer

        curve_model_layers.append(next_layer)

    print("Curve Model Layers: {}".format(curve_model_layers))

    # curve_model = tf.keras.Sequential(curve_model_layers).build((None, 2048))


    class AutoCurve(tf.keras.Model):
        def __init__(self, fix_points=fix_points):
            super(AutoCurve, self).__init__()
            # self.num_classes = curve_model_layers[-1].output_shape[-1]
            self.num_classes = num_classes
            self.fix_points = fix_points
            self.base_model = None
            self._layers = curve_model_layers

        # def build(self, input_shape):
        #     for i, layer in enumerate(curve_model_layers):
        #         setattr(self, "layer_{}".format(i), layer)
        #     return self
        def build(self, input_shape):
            for i, layer in enumerate(curve_model_layers):
                self.layers.append(layer)
            return self

        def __call__(self, inputs, coeffs_t=None, fix_points=None, training=None, *args, **kwargs):
            net = inputs
            # print('AutoCurve.__call__: {}'.format(coeffs_t))
            for layer in self.layers:
                # print(layer)
                # print('AutoCurve.layer-{}: {}'.format(layer, coeffs_t))
                if layer.__class__ in CURVE_LAYERS:
                    # print('AutoCurve.__call__: {}'.format(coeffs_t))
                    net = layer(net, coeffs_t, *args, **kwargs)
                else:
                    net = layer(net, *args, **kwargs)

                # DEBUGGING
                # print("layer: {}, {}".format(layer, net))
                # print("Layer Done.")
            return net

    return AutoCurve().build(input_shape=(None, 2048))
