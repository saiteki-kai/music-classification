import librosa
import numpy as np
from PIL import Image


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
