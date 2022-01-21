import glob
import os

import numpy as np
from PIL import Image

from utils import FEATURES_FOLDER

if __name__ == "__main__":
    path = os.path.join(FEATURES_FOLDER, 'mel', "**", "**", "*.tif")
    files = glob.glob(path)

    # mean, std = np.load("scaling.npy")
    min, max = np.load("scaling.npy")
    print(min, max)

    for i, filepath in enumerate(files):
        img = np.asarray(Image.open(filepath))
        # img = (img - mean) / std
        img = (img - min) / (max - min)
        Image.fromarray(img).save(filepath)
