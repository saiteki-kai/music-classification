"""Split calculated features into folders by genre and subset (training, validation, test). """

import os

from project.config import OUTPUT_FOLDER, DATA_FOLDER, FEATURES_FOLDER, SUBSET
from project.utils.fma import load_tracks

if __name__ == "__main__":
    tracks = load_tracks(DATA_FOLDER, subset=SUBSET)

    for index, row in tracks.iterrows():
        tid_str = f"{index:06d}"
        filename = f"{tid_str[:3]}_{tid_str}.png"

        # source filepath
        filepath = os.path.join(os.path.join(OUTPUT_FOLDER, SUBSET), filename)

        # get the top genre and the split set
        genre_top = str(row["track", "genre_top"]).lower().replace("/", "-")
        split = str(row["set", "split"])

        # destination folder
        folder = os.path.join(FEATURES_FOLDER, SUBSET, split, genre_top)

        # move the file in the features folder
        if os.path.exists(filepath):
            if not os.path.exists(folder):
                os.makedirs(folder)
            os.rename(filepath, os.path.join(folder, filename))
