import os

import librosa
import numpy as np
import pandas as pd

DATA_FOLDER = os.path.join(os.getcwd(), "data")
FMA_RAW = os.path.join(DATA_FOLDER, "raw")
FEATURES_FOLDER = os.path.join(DATA_FOLDER, "features")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "output")


def load_tracks(datafolder, subset=None):
    tracks = pd.read_csv(os.path.join(datafolder, "tracks.csv"), index_col=0, header=[0, 1])
    tracks['track', 'genre_top'] = tracks['track', 'genre_top'].astype('category')

    SUBSETS = ("small", "medium", "large")
    subset_dtype = pd.CategoricalDtype(categories=SUBSETS, ordered=True)
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(subset_dtype)

    if subset is not None:
        if subset in SUBSETS:
            tracks = tracks[tracks['set', 'subset'] <= subset]
        else:
            raise ValueError(f"subset {subset} is not valid")

    return tracks


def compute_mfcc(filepath, duration=None):
    y, sr = librosa.load(filepath, sr=None, mono=True, duration=duration)

    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)

    return np.concatenate((mfcc, mfcc_d, mfcc_dd))


def get_audio_infos(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return sr, librosa.get_duration(y, sr=sr)
