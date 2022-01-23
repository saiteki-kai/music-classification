import matplotlib.pyplot as plt


def plot_history(history):
    x_plot = list(range(1, len(history["loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.plot(x_plot, history["loss"])
    ax1.plot(x_plot, history["val_loss"])
    ax1.legend(["Training", "Validation"])

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.plot(x_plot, history["accuracy"])
    ax2.plot(x_plot, history["val_accuracy"])
    ax2.legend(["Training", "Validation"], loc="lower right")

    fig.show()
