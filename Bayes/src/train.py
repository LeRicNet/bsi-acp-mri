"""Train methods for SWAG model"""
from jupyterplot import ProgressPlot
import tensorflow as tf
import tensorflow_addons as tfa

from .swag2 import SWAG
from .train_utils import train_epoch, evaluate_epoch, plot_model_performance
from .data_manager import ACPMRILite

def train_with_SWAG(
        base_model,
        subspace_type,
        data_dir="acpmrilite",
        num_epochs=100,
        init_lr=tfa.optimizers.TriangularCyclicalLearningRate(1e-5,1e-1, 50),
        swag_init_lr=tfa.optimizers.TriangularCyclicalLearningRate(1e-2,5e-2, 20),
        max_num_swag_models=20,
        swag_start=10,
        swag_c_epochs=1,
        train_verbosity_interval=1,
        eval_frequency=10,
        base_optimizer=tfa.optimizers.SGDW,
        loss_fn="categorical_crossentropy",
        weight_decay=1e-4,
        momentum=0.9,
        base_device="/GPU:0",
        swag_device="/CPU:0",
        checkpoint_dir="./ckpts",
):
    # Load Data: data.train, data.test
    # TODO: Replace with BaseLoader + config
    data = ACPMRILite()
    data.load()

    # Initialize SWAG model from base_model template
    swag_model = SWAG(base_model, subspace_type, subspace_kwargs={'max_rank': max_num_swag_models})

    # Compile the models on separate devices. SWAG does not need to run on GPU...
    with tf.device(swag_device):
        swag_model.compile(optimizer=base_optimizer(weight_decay=weight_decay,
                                                    learning_rate=swag_init_lr,
                                                    momentum=momentum),
                           loss=loss_fn,
                           metrics=[tf.keras.metrics.AUC(curve="PR")]
                           )

    with tf.device(base_device):
        base_model.compile(optimizer=base_optimizer(weight_decay=weight_decay,
                                                    learning_rate=init_lr,
                                                    momentum=momentum),
                           loss=loss_fn,
                           metrics=[tf.keras.metrics.AUC(curve="PR")])

    n_ensembled = 0
    model_performance = []
    progress_plot = ProgressPlot(
        plot_names=["loss", "aupr"],
        line_names=["train", "test", "swag"],
        x_lim=[0, int(num_epochs + 1)],
        y_lim=[-0, 1])

    for epoch in range(num_epochs):

        if epoch % train_verbosity_interval == 0:
            verbose = False
        else:
            verbose = False
        # Train base_model
        base_model, performance_dict = train_epoch(epoch, data,
                                                   base_model, base_device, verbose)

        if epoch % eval_frequency == 0:
            performance_dict = evaluate_epoch(epoch, data, base_model,
                                              base_device, performance_dict, verbose)

        if (epoch + 1) > swag_start and (epoch + 1 - swag_start) % swag_c_epochs == 0:
            n_ensembled += 1
            with tf.device(swag_device):
                swag_model.collect_model(base_model)
            if epoch == 0 or epoch % eval_frequency == eval_frequency - 1 or epoch == num_epochs -1:
                swag_model.set_swa()
                performance_dict = evaluate_epoch(epoch, data, swag_model,
                                                  swag_device, performance_dict, verbose, swag=True)


        # Update Performance and ProgressPlot
        model_performance.append(performance_dict)
        progress_plot = plot_model_performance(model_performance, progress_plot)

    progress_plot.finalize()








