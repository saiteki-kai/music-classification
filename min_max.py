"""Compute min and max of the computed features from the training set."""

import glob
import os

import numpy as np
from PIL import Image

from utils import FEATURES_FOLDER


if __name__ == "__main__":
    path = os.path.join(FEATURES_FOLDER, "mel", "training", "**", "*.tif")
    files = glob.glob(path)

    input_size = (128, 128)

    min = np.Infinity
    max = -np.Infinity
    print(min, max)

    for i, filepath in enumerate(files):
        with Image.open(filepath) as img:
            xi = np.asarray(img)
            xi_min = np.min(xi)
            xi_max = np.max(xi)
            if xi_min < min:
                min = xi_min
            if xi_max > max:
                max = xi_max

    print(min, max)
    np.save("scaling", (min, max))
