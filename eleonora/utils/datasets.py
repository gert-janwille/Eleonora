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

    def _load_KDEF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('v.jpg')):
                    print("file Removed", filename)
                    os.remove(os.path.join(folder, filename))
                if filename.lower().endswith(('h.jpg')):
                    print("file Removed", filename)
                    os.remove(os.path.join(folder, filename))
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        c2 = 1
        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = []
        for file_arg, file_path in enumerate(file_paths):
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]
            try:
                emotion_arg = class_to_arg[file_emotion]
                emotions.append(emotion_arg)
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                faces[file_arg] = image_array
            except:
                continue

            wait = int(0 + (100 - 0) * (c2 - 0) / ((num_faces-1) - 0))
            print(" Initializing Dataset Faces: %s%s%s%s" % (T, wait, "%", W), end="\r")
            c2 = c2 + 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions


def split_data(X, y, validation_split=.2):
    num_train_samples = int((1 - validation_split)*len(X))
    X_train, y_train, X_test, y_test = X[:num_train_samples], y[:num_train_samples], X[num_train_samples:], y[num_train_samples:]
    print("\tTraining Set:", len(X_train), len(y_train))
    print("\tTest Set:", len(X_test), len(y_test))
    return X_train, X_test, y_train, y_test

def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4,
                'surprise':5, 'neutral':6}
    elif dataset_name == 'imdb':
        return {'woman':0, 'man':1}
    elif dataset_name == 'KDEF':
        return {'AN':0, 'DI':1, 'AF':2, 'HA':3, 'SA':4, 'SU':5, 'NE':6}
    else:
        raise Exception('Invalid dataset name')
