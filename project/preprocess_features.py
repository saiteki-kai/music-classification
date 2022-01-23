"""Compute mean and standard deviation of the computed features from the training set."""

import glob
import os

import numpy as np
from PIL import Image

from project.utils import FEATURES_FOLDER, OUTPUT_FOLDER


def compute_running_scaling(files, input_size):
    mean = np.zeros(input_size, dtype=np.float64)
    var = np.zeros(input_size, dtype=np.float64)

    for i, filepath in enumerate(files):
        with Image.open(filepath) as img:
            xi = np.asarray(img)

            new_mean = mean + ((xi - mean) / (i + 1))  # i + 1 = current n
            var = (i * var + (xi - mean) * (xi - new_mean)) / (i + 1)

            mean = new_mean

    return mean, np.sqrt(var)


if __name__ == "__main__":
    path = os.path.join(FEATURES_FOLDER, "small", "**", "**", "*.tif")
    files = glob.glob(path)

    scaling_filepath = os.path.join(OUTPUT_FOLDER, "scaling.npy")

    if not os.path.exists(scaling_filepath):
        train_files = [f for f in files if "training" in f]
        print(len(train_files), len(files))

        mean, std = compute_running_scaling(files, input_size=(13, 2580))
        np.save(scaling_filepath, (mean, std))
    else:
        mean, std = np.load(scaling_filepath)

    # apply scaling

    for i, filepath in enumerate(files):
        img = np.asarray(Image.open(filepath))
        img = (img - mean) / std
        Image.fromarray(img).save(filepath)
