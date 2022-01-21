import os
import glob
import warnings

import librosa
from PIL import Image
from multiprocessing import Pool

from bad_samples import get_ignore_list
from utils import FMA_RAW, OUTPUT_FOLDER, SUBSET, compute_melspectrogram, spectrogram_to_image, split_audio

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    files = glob.glob(os.path.join(FMA_RAW, SUBSET, "**/*.mp3"), recursive=True)

    # remove files with a duration of less than 30s

    ignore_list = get_ignore_list(SUBSET)

    for f in ignore_list:
        f = os.path.join(FMA_RAW, SUBSET, f)
        if f in files:
            files.remove(f)

    print("files remaining: ", len(files))

    def compute(filepath):
        y, sr = librosa.load(filepath, sr=22050, mono=True, duration=29.70)
        segments = split_audio(y, n_segments=1)
        return list(map(lambda s : compute_melspectrogram(s, sr), segments))

    # compute features using a thread pool

    nb_workers = 8 #int(1.5 * len(os.sched_getaffinity(0)))
    with Pool(processes=nb_workers) as pool:
        it = pool.imap(compute, files, chunksize=250)
        for i, data in enumerate(it):
            for k, segment in enumerate(data):
                fp = "_".join(files[i].split("/")[-2:])
                fp = os.path.splitext(fp)[0]
                fp = os.path.join(OUTPUT_FOLDER, SUBSET, f"{fp}_s{k}.png")

                img = spectrogram_to_image(segment)
                img.save(fp)

            if i % 250 == 0:
                print(i)
