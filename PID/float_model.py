import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization, Activation, GlobalAveragePooling1D, Masking, Lambda, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, HeUniform, GlorotUniform
from tensorflow.keras.layers import Dropout, ReLU

# CHARGE_DIV = 3.0289125927999994
CHARGE_DIV = 1

# Define a custom masking function
def custom_masking(inputs, epsilon=1e-5):
    # Check if all attributes are close to zero (not just exactly zero)
    mask = tf.reduce_all(tf.abs(inputs) < epsilon, axis=-1, keepdims=True)
    return tf.where(mask, tf.zeros_like(inputs), inputs)


def relative_squared_error(y_true, y_pred, epsilon=0.1):
    return tf.reduce_mean(tf.square((y_true - y_pred) / (y_true+epsilon)))

def mape_loss(y_true, y_pred, epsilon=0.1):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon)))


def custom_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred)*CHARGE_DIV))

def log_cosh_error():
    return tf.keras.losses.LogCosh()

from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization, Activation, GlobalAveragePooling1D, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, HeUniform, GlorotUniform

def baseline_with_correlation(inputs_shape, output_shape, kernel_size=1, filters=4, neurons_1=64, neurons_2=32, num_heads=4, key_dim=8):
    # Initialize inputs
    inputs = Input(shape=(None, inputs_shape[-1]), name='model_input')  # Allow arbitrary input size
    
    masked_inputs = Lambda(custom_masking, name="custom_mask")(inputs)

    main = BatchNormalization(name='norm_input')(masked_inputs)
    
    # Step 1: Extract first 2 features (coordinates)
    coords = inputs[:, :2, :]  # Get the first two features across all inputs
    
    # Apply Conv1D to learn correlations in the coordinates (first 2 features)
    coords = Conv1D(filters=filters, kernel_size=kernel_size, name='Conv1D_coords', kernel_initializer=HeNormal())(coords)
    coords = Activation('relu')(coords)
    
    # Step 2: Extract the last 2 features
    last_features = inputs[:, -2:, :]  # Get the last two features
    
    # Project the last features to match the feature size of coordinates (e.g., 16 features)
    last_features_projected = Dense(16, activation='relu', kernel_initializer=HeNormal())(last_features)  # Project to match 16 features
    
    # Step 3: Concatenate the correlations of the first 2 features with the last 2 features (now both have 16 features)
    merged = Concatenate(axis=1)([coords, last_features_projected])  # Shape: (None, 2, 16) + (None, 2, 16) = (None, 4, 16)
    
    # Apply Conv1D to the merged representation
    main = Conv1D(filters=filters, kernel_size=kernel_size, name='Conv1D_merged', kernel_initializer=HeNormal())(merged)
    main = Activation('relu')(main)
    
    # Step 4: Aggregation layer (Pooling or Flatten)
    main = GlobalAveragePooling1D(name='avgpool')(main)
    
    # JetID branch, 3-layer MLP for regression output
    jet_id = Dense(neurons_1, name='Dense_1_jetID', kernel_initializer=HeUniform())(main)
    jet_id = Activation('relu', name='relu_1_jetID')(jet_id)
    
    jet_id = Dense(neurons_2, name='Dense_2_jetID', kernel_initializer=HeUniform())(jet_id)
    jet_id = Activation('relu', name='relu_2_jetID')(jet_id)
    
    jet_id = Dense(neurons_2, name='Dense_3_jetID', kernel_initializer=HeUniform())(jet_id)
    jet_id = Activation('relu', name='relu_3_jetID')(jet_id)
    
    # Final output layer (for regression task)
    jet_id = Dense(output_shape[-1], name='Dense_4_jetID', kernel_initializer=GlorotUniform())(jet_id)
    jet_id = Activation('linear', name='jet_id_output')(jet_id)
    
    # Define the model
    model = Model(inputs=inputs, outputs=jet_id)
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(), metrics='mae')
    
    print(model.summary())
    
    return model


def baseline_float(inputs_shape, output_shape, kernel_size=1, filters=4, neurons_1=64, neurons_2=32, num_heads=4, key_dim=8):
    # Initialize inputs
    inputs = Input(shape=(4, inputs_shape[-1]), name='model_input')  # Allow arbitrary input size
    # masked_inputs = Masking(mask_value=0.0)(inputs)
    masked_inputs = Lambda(custom_masking, name="custom_mask")(inputs)
    
    # Main branch
    main = BatchNormalization(name='norm_input')(masked_inputs)
    
    # First Conv1D
    main = Conv1D(filters=filters, kernel_size=kernel_size, name='Conv1D_1', kernel_initializer=HeNormal())(main)
    # main = BatchNormalization(name='c_norm_1')(main)
    main = Activation('relu', name='relu_1')(main)
    
    
    # Second Conv1D
    main = Conv1D(filters=filters, kernel_size=1, name='Conv1D_2', kernel_initializer=HeNormal())(main)
    # main = BatchNormalization(name='c_norm_2')(main)
    main = Activation('relu', name='relu_2')(main)
    
    # Third Conv1D
    main = Conv1D(filters=filters, kernel_size=1, name='Conv1D_3', kernel_initializer=HeNormal())(main)
    # main = BatchNormalization(name='c_norm_3')(main)
    main = Activation('relu', name='relu_3')(main)

    # Permutation-invariant aggregation
    main = GlobalAveragePooling1D(name='avgpool')(main)

    
    # JetID branch, 3-layer MLP
    jet_id = Dense(neurons_1, name='Dense_1_jetID', kernel_initializer=HeUniform())(main)
    # jet_id = BatchNormalization(name='norm_1')(jet_id)
    jet_id = Activation('relu', name='relu_1_jetID')(jet_id)
    
    
    jet_id = Dense(neurons_2, name='Dense_2_jetID', kernel_initializer=HeUniform())(jet_id)
    # jet_id = BatchNormalization(name='norm_2')(jet_id)
    jet_id = Activation('relu', name='relu_2_jetID')(jet_id)
    
    jet_id = Dense(neurons_2, name='Dense_3_jetID', kernel_initializer=HeUniform())(jet_id)
    # jet_id = BatchNormalization(name='norm_3')(jet_id)
    jet_id = Activation('relu', name='relu_3_jetID')(jet_id)
    
    
    jet_id = Dense(output_shape[-1], name='Dense_4_jetID', kernel_initializer=GlorotUniform())(jet_id)
    jet_id = Activation('linear', name='jet_id_output')(jet_id)
    
    # Define the model
    model = Model(inputs=inputs, outputs=jet_id)
    # model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    from tensorflow.keras.optimizers import Adam 
    # model.compile(optimizer=Adam(learning_rate=1e-3),
    #                         loss=tf.keras.losses.Huber(),
    #                         metrics = ['mae', 'mean_squared_error'],
    #                         weighted_metrics = ['mae', 'mean_squared_error'])
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(), metrics='mae')
    
    print(model.summary())
    
    return model


def dense_model(inputs_shape, output_shape):
    inputs = Input(shape=(8,), name='model_input')  # (None, features)
    masked_inputs = Masking(mask_value=0.0)(inputs)  # Ignore padded values
    # x = GlobalAveragePooling1D()(masked_inputs)
    # First Dense layer with BatchNorm and Dropout
    x = Dense(128, kernel_initializer=HeNormal())(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    # Second Dense layer
    x = Dense(64, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)

    # Third Dense layer
    x = Dense(32, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Output layer (sum of charges, theta, phi)
    outputs = Dense(output_shape[-1], activation="linear", kernel_initializer=GlorotUniform())(x)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs, name="VariableLengthDenseModel")
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(), metrics=[custom_mae])

    # Model summary
    model.summary()
    print("OUT SHAPE: ", model.output_shape)

    return model