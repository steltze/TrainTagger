import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# from tagger.train.models import baseline
from float_model import baseline_float, dense_model, CHARGE_DIV, baseline_with_correlation
import utils
from normalize_data import transform_train_set

num_threads = 12
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)

# GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
tf.keras.utils.set_random_seed(420) #not a special number 
BATCH_SIZE = 1024
EPOCHS = 500
VALIDATION_SPLIT = 0.1 # 10% of training set will be used for validation set. 

# Sparsity parameters
I_SPARSITY = 0.0 #Initial sparsity
F_SPARSITY = 0.1 #Final sparsity

def prune_model(model, num_samples):
    """
    Pruning settings for the model. Return the pruned model
    """

    print("Begin pruning the model...")

    #Calculate the ending step for pruning
    end_step = np.ceil(num_samples / BATCH_SIZE).astype(np.int32) * EPOCHS

    #Define the pruned model
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=I_SPARSITY, final_sparsity=F_SPARSITY, begin_step=0, end_step=end_step)}
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    pruned_model.compile(optimizer='adam',
                            loss={'prune_low_magnitude_jet_id_output': 'categorical_crossentropy', 'prune_low_magnitude_pT_output': tf.keras.losses.Huber()},
                            metrics = {'prune_low_magnitude_jet_id_output': 'categorical_accuracy', 'prune_low_magnitude_pT_output': ['mae', 'mean_squared_error']},
                            weighted_metrics = {'prune_low_magnitude_jet_id_output': 'categorical_accuracy', 'prune_low_magnitude_pT_output': ['mae', 'mean_squared_error']})

    print(pruned_model.summary())

    return pruned_model
    
def train_qkeras_model():
    X_train, y_train = utils.prepare_dataset()

    print("Number of samples = ", X_train.shape, y_train.shape)

    KERNEL_SIZE = 1
    FILTERS = 16
    NEURONS_1 = 64
    NEURONS_2 = 32
    # model = baseline_float(X_train.shape, y_train.shape, kernel_size=KERNEL_SIZE, filters=FILTERS, neurons_1=NEURONS_1, neurons_2=NEURONS_2)
    model = baseline_float(X_train.shape, y_train.shape, kernel_size=KERNEL_SIZE, filters=FILTERS, neurons_1=NEURONS_1, neurons_2=NEURONS_2)

    
    if y_train.shape[-1] == 1:
        checkpoint_filepath = f"models/dummy_keras_particle_ks_{KERNEL_SIZE}_f_{FILTERS}_n1_{NEURONS_1}_n2_{NEURONS_2}.h5"
    else:
        checkpoint_filepath = f"models/dummy_keras_phi_thera_charge_ks_{KERNEL_SIZE}_f_{FILTERS}_n1_{NEURONS_1}_n2_{NEURONS_2}.h5"
        
    callbacks = [
                EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode="min", verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5),
                ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_loss", save_best_only=True, save_weights_only=False, mode="min", verbose=1)
                ]
    
    history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=2,
                            # validation_data=(X_test, y_test),
                            validation_split=VALIDATION_SPLIT,
                            callbacks = [callbacks],
                            shuffle=True)

    utils.plot_losses(history, kernel_size=KERNEL_SIZE, filters=FILTERS, neurons_1=NEURONS_1, neurons_2=NEURONS_2, particle=y_train.shape[-1])

if __name__ == "__main__":
    # train_test_model()
    # train_qkeras_model()
  
    X_train, y_train = utils.prepare_dataset()
    utils.predict_and_plot_error_curves("/home/stzelepi/PixESL/my_fork/TrainTagger/PID/models/dummy_keras_phi_thera_charge_ks_1_f_16_n1_64_n2_32.h5" \
        , X_train, y_train)