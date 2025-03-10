import matplotlib.pyplot as plt


def plot_losses(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the training and validation loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('PID/training_loss_plot.png', dpi=300)  # Saves as a high-resolution PNG file
    plt.show()