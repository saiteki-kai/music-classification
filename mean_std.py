"""Compute mean and standard deviation of the computed features from the training set."""

import glob
import os

import numpy as np
from PIL import Image

from utils import FEATURES_FOLDER

if __name__ == "__main__":
    path = os.path.join(FEATURES_FOLDER, 'mel', "training", "**", "*.tif")
    files = glob.glob(path)

    input_size = (128, 128)

    mean = np.zeros(input_size, dtype=np.float64)
    var = np.zeros(input_size, dtype=np.float64)

    for i, filepath in enumerate(files):
        with Image.open(filepath) as img:
            xi = np.asarray(img)

            new_mean = mean + ((xi - mean) / (i + 1))  # i + 1 = current n
            var = (i * var + (xi - mean) * (xi - new_mean)) / (i + 1)

            mean = new_mean

    np.save("scaling_meanstd", (mean, np.sqrt(var)))
