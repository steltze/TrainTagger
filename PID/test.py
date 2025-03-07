import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from tagger.train.models import baseline
import utils

num_threads = 8
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
EPOCHS = 100
VALIDATION_SPLIT = 0.1 # 10% of training set will be used for validation set. 

# Sparsity parameters
I_SPARSITY = 0.0 #Initial sparsity
F_SPARSITY = 0.1 #Final sparsity

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
    
    num_sets = 5000
    max_samples = 20
    num_channels = 10

    X_train, y_train = generate_data(num_sets, max_samples, num_channels)
    
    model = baseline((0, num_channels), (1, 1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])



    model.fit(X_train, y_train, epochs=10, batch_size=32)

    X_test, y_test = generate_data(100, max_samples, num_channels)

    model.evaluate(X_test, y_test)
    
def train_qkeras_model():
    num_channels = 10 
    model = get_test_model(num_channels)

    num_sets = 50000
    max_samples = 20
    num_channels = 10

    X_train, y_train = generate_data(num_sets, max_samples, num_channels)
    X_test, y_test = generate_data(1000, max_samples, num_channels)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    model.evaluate(X_test, y_test)
    
    utils.plot_losses(history)

if __name__ == "__main__":
    train_qkeras_model()