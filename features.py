import glob
import os
import warnings
from multiprocessing import Pool

import numpy as np
from PIL import Image

from bad_samples import get_ignore_list
from utils import FMA_RAW, OUTPUT_FOLDER, compute_mfcc, get_audio_infos

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    files = glob.glob(os.path.join(FMA_RAW, "**/*.mp3"), recursive=True)

    # remove files with a duration of less than 30s

    ignore_list = get_ignore_list("medium")

    for f in ignore_list:
        f = os.path.join(FMA_RAW, f)
        if f in files:
            files.remove(f)

    # remove the files with a sample rate other than 44100

    bad_sr_files = np.load(os.path.join(OUTPUT_FOLDER, "bad_samples_files.npy"))
    print("files with bad sample rate: ", len(bad_sr_files))

    for f in bad_sr_files:
        if f in files:
            files.remove(f)

    print("files remaining: ", len(files))


    def compute(filepath):
        return compute_mfcc(filepath, duration=29.95)


    # compute features using a thread pool

    with Pool(processes=6) as pool:
        it = pool.imap(compute, files, chunksize=250)
        for i, data in enumerate(it):
            try:
                fp = '_'.join(files[i].split('/')[-2:])
                fp = os.path.splitext(fp)[0]
                fp = os.path.join(OUTPUT_FOLDER, "mfcc", f"{fp}.tif")
                Image.fromarray(data).save(fp)

                if i % 500 == 0:
                    print(i)

            except BaseException as e:
                print(i, get_audio_infos(files[i]))
                print(e)
