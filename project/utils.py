import os

import librosa
import numpy as np
import pandas as pd
from PIL import Image

DATA_FOLDER = os.path.abspath(os.path.join("..", "data"))
FMA_RAW = os.path.abspath(os.path.join(DATA_FOLDER, "raw"))
FEATURES_FOLDER = os.path.abspath(os.path.join(DATA_FOLDER, "features"))
OUTPUT_FOLDER = os.path.abspath(os.path.join(DATA_FOLDER, "output"))

SUBSET = "small"


def load_tracks(datafolder, subset=None):
    path = os.path.join(datafolder, "tracks.csv")

    tracks = pd.read_csv(path, index_col=0, header=[0, 1])
    tracks["track", "genre_top"] = tracks["track", "genre_top"].astype("category")

    SUBSETS = ("small", "medium", "large")
    subset_dtype = pd.CategoricalDtype(categories=SUBSETS, ordered=True)
    tracks["set", "subset"] = tracks["set", "subset"].astype(subset_dtype)

    if subset is not None:
        if subset in SUBSETS:
            tracks = tracks[tracks["set", "subset"] <= subset]
        else:
            raise ValueError(f"subset {subset} is not valid")

    return tracks


def load_handcrafted(datafolder):
    path = os.path.join(datafolder, "features.csv")
    features = pd.read_csv(path, index_col=0, header=[0, 1, 2])

    return features


def compute_melspectrogram(y, sr):
    mel = librosa.feature.melspectrogram(y, sr=sr, fmax=8000)
    mel = librosa.power_to_db(mel)
    return mel


def compute_mfcc(y, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def spectrogram_to_image(S):
    S = (S - np.min(S)) / (np.max(S) - np.min(S))
    S = np.uint8(S * 255)
    S = np.flip(S, 0)
    S = 255 - S
    return Image.fromarray(S)


def split_audio(y, n_segments=10):
    samples_for_segment = int(len(y) / n_segments)

    segments = []
    for i in range(n_segments):
        s = y[i * samples_for_segment: i * samples_for_segment + samples_for_segment]
        segments.append(s)

    return segments


def get_audio_infos(filepath):
    """Get the sample rate and duration of a given audio file."""
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return sr, librosa.get_duration(y, sr=sr)
