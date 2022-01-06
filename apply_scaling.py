import glob
import os

import numpy as np
from PIL import Image

from utils import FEATURES_FOLDER

if __name__ == "__main__":
    subset = "small"

    path = os.path.join(FEATURES_FOLDER, subset, "**", "**", "*.tif")
    files = glob.glob(path)

    mean, std = np.load("scaling.npy")

    for i, filepath in enumerate(files):
        img = np.asarray(Image.open(filepath))
        img = (img - mean) / std
        Image.fromarray(img).save(filepath)
