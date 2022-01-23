"""Compute the features

Extract the logarithmic Mel-spectrogram from raw audio files and save as images in the output
folder. It is possible to extract the spectrogram from audio file segments by setting the
N_SEGMENTS parameter.
"""

import glob
import multiprocessing
import os
import warnings

import librosa

from project.bad_samples import get_ignore_list
from project.utils import (
    FMA_RAW,
    OUTPUT_FOLDER,
    SUBSET,
    spectrogram_to_image,
    split_audio,
    compute_melspectrogram
)

warnings.filterwarnings("ignore")


def compute(filepath):
    """Compute the spectrogram for each segment given a filepath."""
    y, sr = librosa.load(filepath, sr=22050, mono=True, duration=29.70)
    segments = split_audio(y, n_segments=N_SEGMENTS)
    return list(map(lambda s: compute_melspectrogram(s, sr), segments))


if __name__ == "__main__":
    N_SEGMENTS = 1  # number of segments to divide audio into
    NB_WORKERS = 8  # number of processes for the features computation

    files = glob.glob(os.path.join(FMA_RAW, SUBSET, "**/*.mp3"), recursive=True)
    files = files[:5]

    # remove files with a duration of less than 30s

    ignore_list = get_ignore_list(SUBSET)

    for f in ignore_list:
        f = os.path.join(FMA_RAW, SUBSET, f)
        if f in files:
            files.remove(f)

    print("files remaining: ", len(files))

    # compute features using a thread pool

    with multiprocessing.Pool(processes=NB_WORKERS) as pool:
        it = pool.imap(compute, files, chunksize=250)
        for i, data in enumerate(it):
            for k, segment in enumerate(data):
                fp = "_".join(files[i].split("/")[-2:])
                fp = os.path.splitext(fp)[0]
                fp = os.path.join(OUTPUT_FOLDER, SUBSET, f"{fp}_s{k}.png")

                img = spectrogram_to_image(segment)
                img.save(fp)

            if i % 250 == 0:
                print(i)  # progress
