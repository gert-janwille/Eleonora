import numpy as np
from keras.utils import np_utils

def preprocess_input(x):
    temp = []
    for idx, arr in enumerate(x):
        arr = np.array(arr).astype('float32')
        arr = arr / 255.0
        temp.append(arr)
    return temp

def to_categorical(classes, num_classes=2):
    temp = []
    for idx, arr in enumerate(classes):
        temp.append(np_utils.to_categorical(arr, num_classes))
    return temp
