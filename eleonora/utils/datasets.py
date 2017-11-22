import os
import cv2
import numpy as np
import pandas as pd
from random import shuffle
from scipy.io import loadmat
from keras.utils import np_utils
from eleonora.common.constants import *
from eleonora.utils.input import message


class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size

        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = './eleonora/data/datasets/imdb_crop/imdb.mat'
        elif self.dataset_name == 'fer2013':
            self.dataset_path = './eleonora/data/datasets/fer2013/fer2013.csv'
        elif self.dataset_name == 'KDEF':
            self.dataset_path = './eleonora/data/datasets/KDEF/'
        else:
            raise Exception('Incorrect dataset name, please input imdb or fer2013')

    def get_data(self):
        message("Initializing " + self.dataset_name + " Dataset")
        if self.dataset_name == 'imdb':
            data = self._load_imdb()
        elif self.dataset_name == 'fer2013':
            data = self._load_fer2013()
        elif self.dataset_name == 'KDEF':
            data = self._load_KDEF()
        return data

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        X = []
        c = 1
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            X.append(face.astype('float32'))
            wait = int(0 + (100 - 0) * (c - 0) / ((len(pixels)-1) - 0))
            print(" Initializing Dataset Faces: %s%s%s%s" % (T, wait, "%", W), end="\r")
            c = c + 1
        X = np.asarray(X)
        X = np.expand_dims(X, -1)
        print("\n")
        y = data['emotion']
        print(T, "\nShape:", len(X), len(y), W)
        return X, y


def split_data(X, y, validation_split=.2):
    num_train_samples = int((1 - validation_split)*len(X))
    X_train, y_train, X_test, y_test = X[:num_train_samples], y[:num_train_samples], X[num_train_samples:], y[num_train_samples:]
    print("\tTraining Set:", len(X_train), len(y_train))
    print("\tTest Set:", len(X_test), len(y_test))
    return X_train, X_test, y_train, y_test
