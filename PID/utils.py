import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

def plot_losses(history, kernel_size=1, filters=4, neurons_1=64, neurons_2=32, particle=1):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']

    # Plot the training and validation loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(mae, label='MAE')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if particle == 1:
        plt.savefig(f'models/training_loss_plot_particle_ks_{kernel_size}_f_{filters}_n1_{neurons_1}_n2_{neurons_2}.png', dpi=300)  # Saves as a high-resolution PNG file
    else:
        plt.savefig(f'models/training_loss_plot_phi_theta_charge_ks_{kernel_size}_f_{filters}_n1_{neurons_1}_n2_{neurons_2}.png', dpi=300)  # Saves as a high-resolution PNG file
    plt.show()

# def validate_model(path_to_model="models/dummy_keras_ks_1_f_4_n1_64_n2_32.h5", path_to_data_X="/home/stzelepi/PixESL/pixesl/ML/X_R=6.2mm,2chip_dump_11_2_0.npy", path_to_data_y="/home/stzelepi/PixESL/pixesl/ML/y_R=6.2mm,2chip_dump_11_2_0.npy"):
    
#     X_test = np.load(path_to_data_X)
#     y_test = np.load(path_to_data_y)
#     model.evaluate(X_test, y_test)

    
def predict_and_plot_error_curves(path_to_model, X_test, y_test):
    model = load_model(path_to_model)
    
    # X_test = np.load(path_to_data_X)
    # y_test = np.load(path_to_data_y)
    
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")

    # Step 2: Generate predictions
    y_pred = model.predict(X_test)

    # Step 3: Calculate individual absolute errors
    absolute_errors = np.abs(y_test - y_pred)

    # Step 2: Define bins and compute mean MAE per bin
    num_bins = 20  # Adjust the number of bins as needed
    # bin_means, bin_edges, _ = binned_statistic(y_test, absolute_errors, statistic='mean', bins=num_bins)

    # Step 3: Compute bin centers for plotting
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Step 4: Plot the results
    plt.figure(figsize=(10, 6))
    plt.hist(absolute_errors, bins=30, edgecolor="black", alpha=0.7)
    # plt.bar(bin_centers, bin_means, width=(bin_edges[1] - bin_edges[0]) * 0.9, alpha=0.7, edgecolor='black')
    plt.xlabel('True Values (binned)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Mean Absolute Error vs. True Values (Binned)')
    plt.grid(axis='y')
    plt.show()
    plt.savefig(f'models/error_curve_phi_theta_charge.png', dpi=300)  # Saves as a high-resolution PNG file
    
    
def prepare_dataset(x_transformer=None, index=None):
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
        
    indices = np.arange(X_train.shape[0])  # Create an index array [0, 1, 2, ..., 149]
    np.random.shuffle(indices)             # Shuffle the indices

    # Step 2: Apply shuffled indices to both arrays

    # X_train[:, :, 0] /= 128
    # X_train[:, :, 1] /= 128
    # X_train[:, :, 2] /= 35_000
    # X_train[:, :, 3] /= CHARGE_DIV
    if not index == None: 
        X_train = X_train[indices][:index]
        y_train = y_train[indices][:index]

    return X_train, y_train
