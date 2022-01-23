"""Define the list of files to be ignored.

More details here:
https://github.com/mdeff/fma/wiki#excerpts-shorter-than-30s-and-erroneous-audio-length-metadata
"""

ignore_list_small = [
    "098/098565.mp3",
    "098/098567.mp3",
    "098/098569.mp3",
    "099/099134.mp3",
    "108/108925.mp3",
    "133/133297.mp3",
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
    "143/143992.mp3",
]


def get_ignore_list(subset):
    """Returns the list to ignore based on the subset of the dataset."""
    if subset == "small":
        ignore_list = ignore_list_small
    else:
        ignore_list = ignore_list_medium

    return ignore_list
