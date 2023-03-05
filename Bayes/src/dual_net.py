import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import time
from jupyterplot import ProgressPlot

class DualNet(tf.keras.Model):

    def __init__(self, base_model):
        super(DualNet, self).__init__()
        self.train_model = tf.keras.models.clone_model(base_model)
        self.test_model = tf.keras.models.clone_model(base_model)
        self.optimizer = base_model.optimizer
        self.test_model.trainable = False

        # loss functions
        self.SFL = tfa.losses.SigmoidFocalCrossEntropy()
        self.TSL = tfa.losses.TripletSemiHardLoss()

        self.training_performance = {
            'Epoch': [],
            'SFL': [],
            'TSL': [],
            'CORAL': [],
            'TrAUPR': [],
            'TeAUPR': [],
            'ValAUPR': []
        }
        self.progress_plot = None

    def call(self, X, y, Xt, yt, training=False):
        y_pred = self.train_model(X, y, training=training)
        yt_pred = self.test_model(Xt, yt, training=training)
        return y_pred, yt_pred

    def train(self, data, epochs, early_stop=True):

        train_data = data.train
        test_data = data.test
        # pp = ProgressPlot(
        #     plot_names=["loss", "aupr"],
        #     line_names=["train", "test", "SFL", "TSL", "CORAL"],
        #     x_lim=[0, int(epochs + 1)])
        if early_stop:
            early_stop = EarlyStop()

        for epoch in range(epochs):
            et_i = time.time()
            epoch_loss_sfl = tf.keras.metrics.Mean()
            epoch_loss_tsl = tf.keras.metrics.Mean()
            epoch_loss_coral = tf.keras.metrics.Mean()
            epoch_tr_auc = tf.keras.metrics.AUC(curve="PR")
            epoch_te_auc = tf.keras.metrics.AUC(curve="PR")
            # for (X, y), (Xt, yt) in zip(train_data, test_data):
            # Split training batches in two
            # train_data = train_data.shuffle()
            for _X, _y in train_data:
                Xt, yt = _X[(int(data.batch_size/2)):], _y[(int(data.batch_size/2)):]
                X, y = _X[:(int(data.batch_size/2))], _y[:(int(data.batch_size/2))]
                # randomly subset data
                # if np.random.choice([0,1], p=[0.7, 0.3]):
                #     X, y = subset(X, y)
                #     Xt, yt = subset(Xt, yt)

                loss_values = self._train_step(X, y, Xt, epoch)
                epoch_loss_sfl.update_state(loss_values[0])
                epoch_loss_tsl.update_state(loss_values[1])
                epoch_loss_coral.update_state(loss_values[2])
                epoch_tr_auc.update_state(*self._eval_train_model(X, y))
                epoch_te_auc.update_state(*self._eval_test_model(Xt, yt))

                self.update_test_weights()

            if epoch % 10 == 0:
                epoch_validation = tf.keras.metrics.AUC(curve="PR")
                for Xv, yv in test_data:
                    epoch_validation.update_state(*self._eval_train_model(Xv, yv))
                print(" *** Validation AUC = {:.3f}".format(epoch_validation.result().numpy()))

            update_dict = {
                'loss': {
                    'SFL': epoch_loss_sfl.result().numpy(),
                    'TSL': epoch_loss_tsl.result().numpy(),
                    'CORAL': epoch_loss_coral.result().numpy()
                },
                'aupr': {
                    'train': epoch_tr_auc.result().numpy(),
                    'test': epoch_te_auc.result().numpy()
                }
            }

            self.training_performance["Epoch"].append(epoch)
            self.training_performance["SFL"].append(epoch_loss_sfl.result().numpy())
            self.training_performance["TSL"].append(epoch_loss_tsl.result().numpy())
            self.training_performance["CORAL"].append(epoch_loss_coral.result().numpy())
            self.training_performance["TrAUPR"].append(epoch_tr_auc.result().numpy())
            self.training_performance["TeAUPR"].append(epoch_te_auc.result().numpy())
            if epoch % 10 == 0:
                self.training_performance["ValAUPR"].append(epoch_validation.result().numpy())
        #     pp.update(update_dict)
        #
        # pp.finalize()
            det = time.time() - et_i
            print("Epoch {}, SFL={:.3f}, TSL={:.3f}, CORAL={:.3f}, TrAUPR={:.3f}, TeAUPR={:.3f} ({:.1} s)".format(
                epoch, epoch_loss_sfl.result(), epoch_loss_tsl.result(),
                epoch_loss_coral.result(), epoch_tr_auc.result(), epoch_te_auc.result(), det))

            if early_stop.check(epoch, epoch_te_auc.result().numpy()):
                print("*** Early Stopping ***")
                break


    @tf.function
    def _train_step(self, X, y, Xt, epoch):
        optimizer = self.optimizer
        with tf.GradientTape() as tape:
            sig_focal_loss = self.SFL(tf.cast(y, tf.float32), self.train_model(X))
            trip_semihard_loss = self.TSL(tf.math.argmax(y, axis=1), self.train_model(X))

            # Dynamic CORAL scale factor
            # if epoch > 30 and epoch % 3 == 0:
            #     scale_factor = 1e4
            # elif epoch <= 30:
            #     scale_factor = 1.
            # else:
            #     scale_factor = 1e2
            scale_factor = 1e3

            coral_loss = self.CORAL(X, Xt, scale_factor)
            loss_values = [sig_focal_loss, trip_semihard_loss, coral_loss]
            loss_values_sum = sig_focal_loss + trip_semihard_loss + coral_loss
            grads = tape.gradient(loss_values_sum, self.train_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.train_model.trainable_variables))
        return loss_values

    def _eval_test_model(self, Xt, yt):
        return (yt, self.test_model(Xt, training=False))

    def _eval_train_model(self, X, y):
        return (y, self.train_model(X, training=False))

    # def CORAL(self, X, Xt, scale_factor=1.):
    #
    #     cov_s_a = tfp.stats.covariance(X)
    #     cov_t_a = tfp.stats.covariance(Xt)
    #     coral_loss_a = tf.reduce_mean(tf.square(tf.subtract(cov_s_a, cov_t_a)))
    #
    #     cov_s_b = tfp.stats.covariance(self.train_model(X))
    #     cov_t_b = tfp.stats.covariance(self.test_model(Xt))
    #     coral_loss_b = tf.reduce_mean(tf.square(tf.subtract(cov_s_b, cov_t_b)))
    #
    #     coral_loss = coral_loss_a + coral_loss_b
    #
    #     return coral_loss_b * scale_factor
    def CORAL(self, X, Xt, scale_factor=1.):

        # cov_s_a = tfp.stats.covariance(X)
        # cov_t_a = tfp.stats.covariance(Xt)
        # coral_loss_a = tf.reduce_mean(tf.square(tf.subtract(cov_s_a, cov_t_a)))

        cov_s_b = tfp.stats.covariance(self.train_model(X))
        cov_t_b = tfp.stats.covariance(self.test_model(Xt))



        return coral_loss_b * scale_factor

    def _init_progress_plot(self, epochs):

        return progress_plot

    def _update_progress_plot(self, progress_plot, sfl_loss, tsl_loss, coral_loss, tr_auc, te_auc):

        return progress_plot

    def update_test_weights(self):
        for i, l in enumerate(self.test_model.layers):
            l.set_weights(self.train_model.layers[i].get_weights())

    def plot_training_performance(self, figsize=(20,20), yscale="log"):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        df = pd.DataFrame(self.training_performance)
        df.plot(x="Epoch", y=["SFL", "TSL", "CORAL"], kind="line", ax=ax1)
        ax1.set_yscale(yscale)
        ax1.set_ylabel("loss")

        df.plot(x="Epoch", y=["TrAUPR", "TeAUPR"], kind="line", ax=ax2)
        ax2.set_ylabel("AUPR")

        plt.show()

    def get_base_model(self):
        return self.train_model



def subset(X, y, subset_percent=0.66):
    random_index = list(range(X.shape[0]))
    np.random.shuffle(random_index)
    random_index = random_index[:int(subset_percent*X.shape[0])]
    return X[random_index], y[random_index]

class EarlyStop(object):

    def __init__(self, min_epoch=10, max_interval=200, threshold_pct=0.01):
        self.min_epoch = min_epoch
        self.max_interval = max_interval
        self.threshold_pct = threshold_pct
        self.top_value = 0.
        self.steps_since_update = 0

    def check(self, epoch, value):
        if epoch >= self.min_epoch:
            if value >= (self.top_value * (1 + self.threshold_pct)):
                self.top_value = value
                self.steps_since_update = 0
            else:
                self.steps_since_update = self.steps_since_update + 1

        if self.steps_since_update > self.max_interval:
            return True
        else:
            return False
