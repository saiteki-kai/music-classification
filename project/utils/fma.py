import os

import pandas as pd


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
