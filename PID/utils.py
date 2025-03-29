import matplotlib.pyplot as plt


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