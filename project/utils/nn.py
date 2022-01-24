import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time


def plot_history(history, save=False, filepath=None):
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

    if save and filepath is not None:
        fig.savefig(filepath)


def prediction_time(model, input_shape, device_name="/cpu:0"):
    dummy_example = np.random.randn(1, *input_shape)
    with tf.device(device_name):
        times = []
        for i in range(10):
            start = time.perf_counter()
            model.predict(dummy_example, batch_size=32)
            end = time.perf_counter() - start
            times.append(end)
        times = np.asarray(times)
        print(np.mean(times) * 1000)
