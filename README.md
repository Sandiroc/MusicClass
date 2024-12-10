# Music Genre Classification with CNN-LSTM

### Sandilya Bhamidipati

This project demonstrates how to perform music genre classification using a **Convolutional Neural Network (CNN)** combined with **Long Short-Term Memory (LSTM)** layers. The model is trained on the **GTZAN dataset**, a popular benchmark dataset for music genre classification tasks.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Preset Model](#Preset)
  - [Custom Model](#Custom)
  - [Classification of New Tracks](#predicting-new-tracks)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Original Study](#original-study)

## Introduction

This project focuses on classifying music tracks into 10 different genres using a **CNN-LSTM** hybrid model. The model leverages the **Mel Frequency Cepstral Coefficients (MFCC)** of audio files as features for classification. The architecture consists of multiple convolutional layers for spatial feature extraction and LSTM layers for capturing temporal dependencies in the sample music.

## Dataset

The model is trained on the **GTZAN dataset**, which consists of 1,000 30-second music tracks (100 tracks per genre) spanning 10 genres:
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

The dataset is divided into:
- **Training set**: 70% of the data
- **Validation set**: 15% of the data
- **Test set**: 15% of the data

The audio files are pre-processed into **MFCCs** to extract features suitable for training a deep learning model.

## Installation

### Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/MusicClass.git
cd MusicClass
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
Make sure that Keras, TensorFlow, Scikit-Learn, and MatPlotLib are installed.

### Configure Data
The GTZAN dataset was obtained from this link:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download.

Navigate to the root directory of this repository and run this command:
```bash
mkdir data
```

Unzip the compressed file from the previous link into the new ```data``` folder. Make sure that ```data``` is the parent directory of these subdirectories & files: ```genres_original```, ```images_original```, ```features_3_sec.csv```, and ```features_30_sec.csv```.


## Usage
Navigate to the root directory of this repository.
### Preset
Using the default trained model, you can run the following command to evaluate performance and generate visualizations. 
```bash
python main.py --save
```
This will:
1. Load the pre-trained model
2. Display accuracy and loss vs epochs plot
3. Display the confusion matrix for model performance on the test set

\
You can run:
```bash
python main.py
```
to directly display these visualizations

### Custom
There is already a pre-trained model that illustrates the efficiency of the deep learning architecture. In case you want to alter the hyperparameters of the model, you can run the following command:
```bash
python main.py --retrain
```

This will:
1. Train the model from scratch with your custom hyperparameters with the GTZAN dataset.
2. Replace the existing trained model with the new one for future use.

You can modify the hyperparameters in ```models/cnn_lstm.py```and ```scripts/train_cnn_lstm.py```

### Predicting New Tracks
**TODO**


## Model Architecture
This deep learning architecture consists of:
1. Convolutional Layers:
    * These layers extract spatial features from the MFCC spectrograms of the audio.
2. Long-Short-Term-Memory Layers (LSTM):
    * These layers capture temporal dependencies in the audio data, which will be useful in capturing the sequential nature of the music.

## Original Study
A link to the original study that detailed this architecture and method can be found here:
https://ieeexplore.ieee.org/document/9997961
