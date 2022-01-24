# Music Genre Classification
The aim of the project is the classification of music genres from audio files.

The `dataset` used is the following: [Click here](https://github.com/mdeff/fma)

We have `pre-processed` the dataset and you can find it [here](https://www.kaggle.com/giuseppemagazz/fma-mel-new)


## Models
This folder contains the trained neural network models.

## Report
This folder contains the report, written in latex, which describes the project and the tasks performed.

## Code
The `notebooks` folder contains the .ipynb files while the `project` contains the python project with the utility functions and features extraction and processing.

"CNN_audio.ipynb" is used to train a CNN like the one described in the paper by "Daniel Kostrzewa".

"CNN_tuning.ipynb" is used to tune, with Hyperband, a CNN.

"analysis.ipynb" contains the EDA for the raw audio file and the mel spectrograms.

"handcraft_nn.ipynb" is used to train and evaluate the neural network for the handcrafted features.

"features_extraction.ipynb" is used to extract features from a specific layer and evaluate using different classifiers.
