import os
import sys
import random
import numpy as np
from eleonora.common.constants import *

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers import Activation
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

batch_size = 128
num_classes = 7
epochs = 10

def modelToJSON(model):
    print("\n[" + T + "*" + W + "] Saving Model...\n")
    # serialize model to JSON
    model_json = model.to_json()
    with open("./eleonora/data/models/cnn_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("./eleonora/data/models/model.h5")
    print("[" + T + "+" + W + "] Model is saved in folder models\n")

def convolutional_neural_network(training_images, test_images, y_train, y_test, img_rows, img_cols, save=False):
    input_shape = (img_rows, img_cols,1)
    print(K.image_data_format())

    showImages(5, training_images, y_train)

    # Reshape Dataset
    # FIXME: image can't be displayed when doing this
    training_images = np.array(training_images).reshape((len(training_images),img_rows, img_cols,1))
    test_images = np.array(test_images).reshape((len(test_images),img_rows, img_cols,1))

    # Normalisatie
    # FIXME: Images turn green after doing this
    training_images = np.array(training_images, dtype=np.float32)
    test_images = np.array(test_images, dtype=np.float32)
    training_images /= 255
    test_images /= 255


    # Building the Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(), metrics=['accuracy'])


    if input("[" + R + "!" + W + "] Start Training? (y/[N]) ") == "y":
        
        # Train the CNN
        print("\n[" + T + "*" + W + "] Start Training")
        model.fit(training_images, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        print("[" + T + "+" + W + "] Done Training\n")

        # Print the accuracy score
        score = model.evaluate(test_images, y_test, verbose=1)
        print("[" + T + "!" + W + "] Accuraty Score of: " + B + str(score[1]*100)+ "%" + W + "\n")

        if save:
            modelToJSON(model)
