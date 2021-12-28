import glob
import os
import warnings
from multiprocessing import Pool

import numpy as np
from PIL import Image

from utils import FMA_RAW, OUTPUT_FOLDER, compute_mfcc, get_audio_infos

warnings.filterwarnings('ignore')

subset = "medium"  # change this

if __name__ == "__main__":
    files = glob.glob(os.path.join(FMA_RAW, "**/*.mp3"), recursive=True)

    # remove files with a duration of less than 30s
    # https://github.com/mdeff/fma/wiki#excerpts-shorter-than-30s-and-erroneous-audio-length-metadata

    ignore_list_small = [
        "098/098565.mp3",
        "098/098567.mp3",
        "098/098569.mp3",
        "099/099134.mp3",
        "108/108925.mp3",
        "133/133297.mp3"
    ]

    ignore_list_medium = [
        "001/001486.mp3",
        "005/005574.mp3",
        "065/065753.mp3",
        "080/080391.mp3",
        "098/098558.mp3",
        "098/098559.mp3",
        "098/098560.mp3",
        "098/098565.mp3",
        "098/098566.mp3",
        "098/098567.mp3",
        "098/098568.mp3",
        "098/098569.mp3",
        "098/098571.mp3",
        "099/099134.mp3",
        "105/105247.mp3",
        "108/108924.mp3",
        "108/108925.mp3",
        "126/126981.mp3",
        "127/127336.mp3",
        "133/133297.mp3",
        "143/143992.mp3"
    ]

    if subset == "small":
        ignore_list = ignore_list_small
    else:
        ignore_list = ignore_list_medium

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
