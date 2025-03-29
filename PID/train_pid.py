import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
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

def get_test_model(num_channels):

    # Input: A set of samples (None means arbitrary number of samples per set)
    inputs = keras.Input(shape=(None, num_channels))  # Variable-length inputs
    masked_inputs = layers.Masking(mask_value=0.0)(inputs)  # Ignore padded values

    # Process the set with Phi
    phi = layers.TimeDistributed(layers.Dense(32, activation="relu"))(masked_inputs)
    phi = layers.TimeDistributed(layers.Dense(64, activation="relu"))(phi)

    # Sum over non-masked elements
    phi_sum = tf.reduce_sum(phi, axis=1)

    # ρ: Apply final MLP
    rho = layers.Dense(64, activation="relu")(phi_sum)
    rho = layers.Dense(32, activation="relu")(rho)
    outputs = layers.Dense(1, activation="linear")(rho)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.summary()
    return model

def generate_data(num_sets, max_samples, num_channels):
    X = []
    y = []
    
    for _ in range(num_sets):
        num_samples = np.random.randint(1, max_samples + 1)
        set_samples = np.random.randn(num_samples, num_channels)
        
        # Define the function: sum all elements in the set
        target = np.sum(set_samples)  # Regression task
        
        X.append(set_samples)
        y.append(target)
    
    # Convert y to numpy array
    y = np.array(y).reshape(-1, 1)
    
    # Pad sequences for batch training
    X = keras.preprocessing.sequence.pad_sequences(X, padding="post", dtype="float32")
    
    return X, y

def train_test_model():
    num_channels = 10    
    
    num_sets = 500000
    max_samples = 20
    num_channels = 10

    X_train, y_train = generate_data(num_sets, max_samples, num_channels)
    
    model = get_test_model(num_channels)
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    X_test, y_test = generate_data(100, max_samples, num_channels)

    model.evaluate(X_test, y_test)

def save_data(X_train, y_train, X_test, y_test):
    np.save("PID/X_train.npy", X_train) 
    np.save("PID/y_train.npy", y_train)
    np.save("PID/X_test.npy", X_test)
    np.save("PID/y_test.npy", y_test)
    
def load_data():
    X_train = np.load("PID/X_train.npy") 
    y_train = np.load("PID/y_train.npy")
    X_test = np.load("PID/X_test.npy")
    y_test = np.load("PID/y_test.npy")
    
    return X_train, y_train, X_test, y_test
    
def train_qkeras_model():
    num_channels = 10 

    num_sets = 500000
    max_samples = 20
    num_channels = 4
    
    # X_train, y_train, X_test, y_test = load_data()
    data_path = "/eos/home-s/stzelepi/PixESL/Datasets/"
    X_train = np.load(data_path + "filtered_phi_thera_charge_inputs_1.npy")[:, :4, :]
    # print(X_train[:3])
    # X_train[:, :, 0] = X_train[:, :, 0]/128
    # X_train[:, :, 1] = X_train[:, :, 1]/128
    # X_train[:, :, 2] = X_train[:, :, 2]/X_train[:, :, 2].mean()
    # X_train[:, :, 3] = X_train[:, :, 3]/X_train[:, :, 3].mean()
    
    y_train = np.load(data_path + "filtered_phi_thera_charge_outputs_1.npy")[:, 1:]
    # y_train[:, 0] = y_train[:, 0]/y_train[:, 0].mean()
    # y_train[:, 1] = y_train[:, 1]/y_train[:, 1].mean()


    
    for index in range(2, 5):
        file_x = f"filtered_phi_thera_charge_inputs_{index}.npy"
        file_y = f"filtered_phi_thera_charge_outputs_{index}.npy"
        X = np.load(data_path + file_x)[:, :4, :]
        y = np.load(data_path + file_y)[:, 1:]
        X_train = np.vstack((X_train, X))
        y_train = np.vstack((y_train, y))
    
    # X_train = transform_train_set(X_train)

    indices = np.arange(X_train.shape[0])  # Create an index array [0, 1, 2, ..., 149]
    np.random.shuffle(indices)             # Shuffle the indices

    # Step 2: Apply shuffled indices to both arrays

    # X_train[:, :, 0] /= 128
    # X_train[:, :, 1] /= 128
    # X_train[:, :, 2] /= 35_000
    # X_train[:, :, 3] /= CHARGE_DIV
    X_train = X_train[indices][:100000]
    y_train = y_train[indices][:100000]

    print("Number of samples = ", X_train.shape, y_train.shape)

    KERNEL_SIZE = 1
    FILTERS = 32
    NEURONS_1 = 128
    NEURONS_2 = 64
    # model = baseline_float(X_train.shape, y_train.shape, kernel_size=KERNEL_SIZE, filters=FILTERS, neurons_1=NEURONS_1, neurons_2=NEURONS_2)
    model = baseline_float(X_train.shape, y_train.shape, kernel_size=KERNEL_SIZE, filters=FILTERS, neurons_1=NEURONS_1, neurons_2=NEURONS_2)

    
    if y_train.shape[-1] == 1:
        checkpoint_filepath = f"models/dummy_keras_particle_ks_{KERNEL_SIZE}_f_{FILTERS}_n1_{NEURONS_1}_n2_{NEURONS_2}.h5"
    else:
        checkpoint_filepath = f"models/dummy_keras_phi_thera_charge_ks_{KERNEL_SIZE}_f_{FILTERS}_n1_{NEURONS_1}_n2_{NEURONS_2}.h5"
        
    callbacks = [
                # EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode="min", verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5),
                # ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_loss", save_best_only=True, save_weights_only=False, mode="min", verbose=1)
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

def validate_model(path_to_model="models/dummy_keras_ks_1_f_4_n1_64_n2_32.h5", path_to_data_X="/home/stzelepi/PixESL/pixesl/ML/X_R=6.2mm,2chip_dump_11_2_0.npy", path_to_data_y="/home/stzelepi/PixESL/pixesl/ML/y_R=6.2mm,2chip_dump_11_2_0.npy"):
    model = load_model(path_to_model)
    X_test = np.load(path_to_data_X)
    y_test = np.load(path_to_data_y)
    model.evaluate(X_test, y_test)    

if __name__ == "__main__":
    # train_test_model()
    train_qkeras_model()
    # validate_model()