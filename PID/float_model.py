import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization, Activation, GlobalAveragePooling1D, Masking
from tensorflow.keras.models import Model

def baseline_float(inputs_shape, output_shape):
    # Define a dictionary for common arguments
    common_args = {
        'kernel_initializer': 'lecun_uniform'
    }

    # Initialize inputs
    inputs = Input(shape=(None, inputs_shape[-1]), name='model_input')  # Allow arbitrary input size
    masked_inputs = Masking(mask_value=0.0)(inputs)
    
    # Main branch
    main = BatchNormalization(name='norm_input')(masked_inputs)
    
    # First Conv1D
    main = Conv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = Activation('relu', name='relu_1')(main)
    
    # Second Conv1D
    main = Conv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = Activation('relu', name='relu_2')(main)
    
    # Linear activation (no quantization, replaced with normal ReLU)
    # main = Activation('relu', name='act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)
    
    # JetID branch, 3-layer MLP
    jet_id = Dense(32, name='Dense_1_jetID', **common_args)(main)
    jet_id = Activation('relu', name='relu_1_jetID')(jet_id)
    
    jet_id = Dense(16, name='Dense_2_jetID', **common_args)(jet_id)
    jet_id = Activation('relu', name='relu_2_jetID')(jet_id)
    
    jet_id = Dense(output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
    jet_id = Activation('linear', name='jet_id_output')(jet_id)
    
    # Define the model
    model = Model(inputs=inputs, outputs=jet_id)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    
    print(model.summary())
    
    return model
