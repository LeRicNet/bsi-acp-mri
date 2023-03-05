import argparse
import datetime
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.util.tf_export import keras_export

# Local Imports
from Bayes.src.data_manager import ACPMRILite
from Bayes.src.train_utils import train_epoch
from Bayes.src.subspaces import PCASpace
from Bayes.src.swag2 import SWAG
from Bayes.src.mode_connectivity import curves, models
from Bayes.src.mode_connectivity.utils import LRSchedule


# Argument Parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./curve/', metavar='DIR',
                        help='training directory (default: /tmp/curve/)')
    parser.add_argument("--data", choices=['MRI', 'CT'], default='MRI',
                        help="String indicating which preloaded dataset to evaluate. [MRI, CT]")
    parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                        help='model name (default: None)')
    parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')
    parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init start point (default: None)')
    parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                        help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init end point (default: None)')
    parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                        help='fix end point (default: off)')
    parser.set_defaults(init_linear=True)
    parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                        help='turns off linear initialization of intermediate points (default: on)')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument("--wd", type=float,
                        help="weight decay")
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum parameter for optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args, _ = parser.parse_known_args()

    return args

def save_run_parameters(args):
    # Check if directory exists
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

@keras_export('keras.optimizers.LazyAdamW')
class LazyAdamW(tfa.optimizers.DecoupledWeightDecayExtension, tfa.optimizers.LazyAdam):
    def __init__(self, weight_decay, *args, **kwargs):
        super(LazyAdamW, self).__init__(weight_decay, *args, **kwargs)


# Main Execution Script
def run(args):

    # Set Random Seed
    # NOTE: Even by setting the random seed we still observe ambiguity in the performance.
    tf.random.set_seed(args.seed)

    # Load Data
    if args.data == "MRI":
        data = ACPMRILite()
        data.batch_size = 2
        data.max_n = 500  # this is max N per class...
        data.load()

    elif args.data == "CT":
        raise ValueError("CT dataset not currently implemented.")

    with tf.device('/CPU:0'):
        base_model = tf.keras.models.load_model(args.init_start, compile=True,
                                            custom_objects={'LazyAdamW': LazyAdamW,
                                                            'PCASpace': PCASpace})
        base_model.load_weights(os.path.join(args.init_start, "cp.ckpt"))
    # base_model.compile(optimizer=LazyAdamW(args.wd))
    print("Point A:\n")
    print(base_model.layers[0].summary())

    autocurve = models.generate_curve_model(base_model.layers[0], args.fix_start,
                                            args.fix_end, args.num_bends)
    # autocurve(np.ndarray((data.batch_size, 2048)), None, None)
    # print(autocurve.summary())
    architecture = getattr(models, args.model)

    with tf.device('/GPU:0'):
        if args.curve is None:
            model = architecture.base(num_classes=num_classes) # num_classes?
        else:
            curve = getattr(curves, args.curve)
            model = curves.CurveNet(
                num_classes=2,
                curve=curve,
                architecture=autocurve,
                num_bends=args.num_bends,
                fix_start=args.fix_start,
                fix_end=args.fix_end,
            )

    # test pass to initialize model weights
    test_aupr = tf.keras.metrics.AUC(curve="PR")
    for X, y in data.test:
        preds = base_model(X)
        test_aupr.update_state(y, preds)
    print("Test AUPR (init): {:.4f}".format(test_aupr.result()))

    # print("Point A")
    model.import_base_parameters(base_model.layers[0], 0)
    # print("\nPoint B")

    with tf.device('/CPU:0'):
        base_model = tf.keras.models.load_model(args.init_end, compile=True,
                                                custom_objects={'LazyAdamW': LazyAdamW,
                                                                'PCASpace': PCASpace})
        base_model.load_weights(os.path.join(args.init_end, "cp.ckpt"))
    # base_model.compile(optimizer=LazyAdamW(args.wd))
    # print("Point B:\n")
    # print(base_model.summary())

    model.import_base_parameters(base_model.layers[0], args.num_bends - 1)

    if args.init_linear:
        print("Linear Initialization")
        # raise NotImplementedError("init_linear() not yet implemented.")
        model.init_linear()

    lr = LRSchedule(
        initial_learning_rate=args.lr,
        total_epochs=args.epochs,
        batch_size=data.batch_size,
        cardinality=int(data.max_n * data.num_classes)
    )
    wd = tf.keras.experimental.CosineDecayRestarts(args.wd, 20)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # optimizer = tf.keras.optimizers.SGD(
    #     learning_rate=1e-4,
    #     momentum=args.momentum
    # )
    # optimizer = tf.keras.optimizers.Adam(lr)
    # optimizer = LazyAdamW(weight_decay=args.wd, learning_rate=lr)

    optimizer = tfa.optimizers.SGDW(weight_decay=wd, learning_rate=lr, momentum=args.momentum)
    # Extend using SWA. This affords the `assign_average_vars` method.
    # optimizer = tfa.optimizers.SWA(optimizer, start_averaging=20)

    # Compile Model
    model.compile(optimizer=optimizer, loss=loss)
    print("Model Compiled.")

    # print("\nLayers:")
    # for layer in model.layers:
    #     print(layer)
    #     print(layer.get_weights())
    # print()
    # print(model(np.zeros((2, 2048))))

    # Summary Stats (Tensorboard)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './modeconnlogs/gradient_tape/' + current_time + '/train'
    test_log_dir = './modeconnlogs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # TRAIN!!!
    epoch_loss = tf.keras.metrics.Mean()
    epoch_aupr = tf.keras.metrics.AUC(curve="PR")
    # print("\nmodel.variables")
    # for var in model.variables:
    #     print("\t{}, {}".format(var.name, var.shape))
    #
    # print("\nmodel.trainable_variables")
    # for var in model.trainable_variables:
    #     print("\t{}, {}".format(var.name, var.shape))

    max_aupr = 0.
    max_epoch = 0
    steps_since_max = 0
    print("Training...")
    with tf.device("/GPU:0"):
        for epoch in range(args.epochs):
            prog_str = 'Epoch {}/{}: '.format(epoch+1, args.epochs)
            for X, y in data.train:
                with tf.GradientTape() as tape:
                    y_hat, l2_loss, curve_weights = model(X)
                    loss_value = loss(y, y_hat) + l2_loss
                    # loss_value = loss(y, model(X))
                    grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss.update_state(loss_value)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_loss.result(), step=epoch)
                tf.summary.histogram('curve_weights', curve_weights, step=epoch)
                for var in model.variables:
                    if 'weight' in var.name or 'bias' in var.name:
                        tf.summary.histogram(var.name, var, step=epoch)




            prog_str = prog_str + 'Loss: {:.5f}'.format(epoch_loss.result())
            # print("Loss {}/{}: {:.4f}".format(epoch+1, args.epochs, epoch_loss.result()))

            if epoch % 1 == 0:
                for X, y in data.test:
                    preds, _, _ = model(X, training=False)
                    epoch_aupr.update_state(y, preds)
                prog_str = prog_str + ', teAUPR: {:.4f}; Max/Since: {}/{}'.format(epoch_aupr.result(), max_aupr,
                                                                                  epoch - max_epoch)
                with train_summary_writer.as_default():
                    tf.summary.scalar('test_aupr', epoch_aupr.result(), step=epoch)

                if epoch_aupr.result() > max_aupr:
                    max_aupr = epoch_aupr.result()
                    model.save_weights("/data/p720/Bayes/notebooks/saved_models/modeconn/best-cp.ckpt")
                    steps_since_max = 0
                    max_epoch = epoch + 1
                else:
                    steps_since_max += 1

                # print("Test AUPR: {:.4f}".format(epoch_aupr.result()))

            print(prog_str)
            # if steps_since_max > 50:
            #     print("Early Stopping. Max AUPR: {:.3f} at Epoch: {}".format(max_aupr, max_epoch))
            #     break
    #
        model.save_weights("/data/p720/Bayes/notebooks/saved_models/modeconn/epoch{}-cp.ckpt".format(epoch))
        model.save("/data/p720/Bayes/notebooks/saved_models/modeconn", include_optimizer=False)
    #
    #
    # print("\nLayers:")
    # for layer in model.layers:
    #     print(layer)
    #     print(layer.get_weights())
    # print()
    #


            # print(model(X))
        #     epoch_loss.update_state(_train_step(model, X, y, loss, optimizer))
        # print(epoch_loss.result())
            # print(loss)
        # train_epoch(epoch=epoch,
        #             data=data,
        #             model=model,
        #             device=None,
        #             performance_dict=None,
        #             loss=loss,
        #             verbose=True)



    # base_model = None
    # if args.resume is None:
    #     for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
    #         print(" ** Index: {}".format(k))
    #         if path is not None:
    #             if base_model is None:
    #                 base_model = tf.keras.models.load_model(path, compile=False)
    #                 curve = models.generate_curve_model(base_model, args.fix_start,
    #                                                     args.fix_end, args.num_bends)
    #                 print(curve.layers)
                # model.import_base_parameters(base_model, k)
    #
    #     if args.init_linear:
    #         print("Linear Initialization")
    #         model.init_linear()

# @tf.function
# def _train_step(model, X, y, loss, optimizer):
#     with tf.GradientTape() as tape:
#         loss_value = loss(y, model(X), from_logits=True)
#         grads = tape.gradient(loss_value, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return loss_value

if __name__ == '__main__':
    # Partition GPU resources
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load command arguments
    args = get_args()
    # Export command arguments for reproducibility
    save_run_parameters(args)

    run(args)
