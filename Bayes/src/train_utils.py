"""Training Loop Utilities"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

def train_epoch(epoch, data, model, device, performance_dict, loss=None, verbose=False):

    # need loss function
    # if loss is None:
    #     loss = tf.keras.losses.get(model.loss)
    # elif issubclass(loss, tf.keras.losses):
    #     pass
    # else:
    #     raise ValueError("Please provide appropriate loss function. "
    #                      "This can be a subclass of tf.keras.losses or "
    #                      "as part of a compiled tf.keras.Model.")
    loss = tf.keras.losses.get(model.loss)

    # Performance Reporting
    epoch_loss = tf.keras.metrics.Mean()
    epoch_aupr = tf.keras.metrics.AUC(curve="PR")

    # Training only
    with tf.device(device):
        for X, y in data.train:
            loss_value = _train_step(model, X, y, loss)

            # Update Performance Metrics
            epoch_loss.update_state(loss_value)
            epoch_aupr.update_state(y, model(X))

    performance_dict = {
        "epoch": epoch,
        "TrLoss": epoch_loss.result().numpy(),
        "TrAUPR": epoch_aupr.result().numpy(),
        "TeLoss": None,
        "TeAUPR": None,
        "SWAGLoss": 0.,
        "SWAGAUPR": 0.
    }

    if verbose:
        print("TrLoss: {:.4f}, TrAUPR: {:.4f}".format(epoch_loss.result().numpy(),
                                                epoch_aupr.result().numpy()))
    return model, performance_dict

def evaluate_epoch(epoch, data, model, device, performance_dict, verbose=False, swag=False):
    # need loss function
    loss = tf.keras.losses.get(model.loss)

    # Performance Reporting
    epoch_loss = tf.keras.metrics.Mean()
    epoch_aupr = tf.keras.metrics.AUC(curve="PR")

    # Training only
    with tf.device(device):
        for X, y in data.test:
            loss_value, preds_and_gt = _eval_step(model, X, y, loss)

            # Update Performance Metrics
            epoch_loss.update_state(loss_value)
            epoch_aupr.update_state(*preds_and_gt)

    assert performance_dict["epoch"] == epoch

    if swag:
        performance_dict["SWAGLoss"] = epoch_loss.result().numpy()
        performance_dict["SWAGAUPR"] = epoch_aupr.result().numpy()
    else:
        performance_dict["TeLoss"] = epoch_loss.result().numpy()
        performance_dict["TeAUPR"] = epoch_aupr.result().numpy()


    if verbose:
        if not swag:
            print("TeLoss: {:.4f}, TeAUPR: {:.4f}".format(epoch_loss.result().numpy(),
                                                          epoch_aupr.result().numpy()))
        else:
            print("SWAGLoss: {:.4f}, SWAGAUPR: {:.4f}".format(epoch_loss.result().numpy(),
                                                              epoch_aupr.result().numpy()))
    return performance_dict

@tf.function
def _train_step(model, X, y, loss):
    opt = model.optimizer
    with tf.GradientTape() as tape:
        loss_value = loss(y, model(X), from_logits=True)
        grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

def _eval_step(model, X, y, loss):
    return loss(y, model(X, training=False), from_logits=True), (y, model(X, training=False))

def plot_model_performance(model_performance, progress_plot):
    update_dict =  {
            'loss': {'train': model_performance[-1]["TrLoss"],
                     'test': [x["TeLoss"] for x in model_performance if x["TeLoss"] is not None][-1],
                     'swag': [x["SWAGLoss"] for x in model_performance if x["SWAGLoss"] is not None][-1]},
            'aupr': {'train': model_performance[-1]["TrAUPR"],
                     'test': [x["TeAUPR"] for x in model_performance if x["TeAUPR"] is not None][-1],
                     'swag': [x["SWAGAUPR"] for x in model_performance if x["SWAGAUPR"] is not None][-1]}
        }

    # if model_performance[-1]["TeLoss"] is not None \
    #         and model_performance[-1]["TeAUPR"] is not None:
    #     update_dict["loss"]["test_loss"] = model_performance[-1]["TeLoss"]
    #     update_dict["aupr"]["test_aupr"] = model_performance[-1]["TeAUPR"]

    progress_plot.update(update_dict)
    return progress_plot

def set_weights(model, vector, device):
    offset = 0
    with tf.device(device):
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if len(layer_weights) > 1:
                layer_weight_shape = layer_weights[0].shape
                num_elements = np.prod(list(layer_weight_shape))
                new_weights = vector.numpy()[offset:(offset + num_elements)]
                new_weights = new_weights.reshape(*layer_weight_shape)

                if layer.__class__ == tf.keras.layers.BatchNormalization:
                    layer.set_weights([new_weights, layer_weights[1], layer_weights[2], layer_weights[3]])
                else:
                    layer.set_weights([new_weights, layer_weights[1]])

                offset += num_elements