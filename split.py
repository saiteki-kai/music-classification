"""Split calculated features into folders by genre and subset (training, validation, test). """

import os

import pandas as pd

from utils import OUTPUT_FOLDER, DATA_FOLDER, FEATURES_FOLDER

if __name__ == "__main__":
    tracks = pd.read_csv(os.path.join(DATA_FOLDER, "tracks.csv"), index_col=0, header=[0, 1])
    tracks['track', 'genre_top'] = tracks['track', 'genre_top'].astype('category')

    # sort and filter the subsets
    subset_dtype = pd.CategoricalDtype(categories=('small', 'medium', 'large'), ordered=True)
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(subset_dtype)
    tracks = tracks[tracks['set', 'subset'] <= 'medium']

    for index, row in tracks.iterrows():
        tid_str = f"{index:06d}"
        filename = f"{tid_str[:3]}_{tid_str}.tif"

        # source filepath
        filepath = os.path.join(os.path.join(OUTPUT_FOLDER, "mfcc"), filename)

        # get the top genre and the split set
        genre_top = str(row['track', 'genre_top']).lower().replace("/", "-")
        split = str(row['set', 'split'])

        # destination folder
        folder = os.path.join(FEATURES_FOLDER, "mfcc", split, genre_top)

        # move the file in the features folder
        if os.path.exists(filepath):
            if not os.path.exists(folder):
                os.makedirs(folder)
            os.rename(filepath, os.path.join(folder, filename))
