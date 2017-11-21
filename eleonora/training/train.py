import os
import sys
import time
import numpy as np
import eleonora.common.ui as ui
from eleonora.common.constants import *

import eleonora.training.modules.dataset_loader as dl
import eleonora.training.modules.cnn as cnn


img_rows, img_cols = 200, 200
SPLIT_PERCENT = 80
saveModel = False
tst = "ok"

def printHeader(headline):
    print(B + T + '\nEleonora' + W + " - " + headline + "\n\n")


def main(args):
    global saveModel

    ui.clearScreen()
    print ('[' + T + '*' + W + '] Starting TRAINING Eleonora %s at %s' %(VERSION, time.strftime("%Y-%m-%d %H:%M")))

    # Prepare the Dataset
    training_images, test_images, y_train, y_test = dl.loadDataset("./eleonora/training/train_data/faces/_processed/", img_rows, img_cols, SPLIT_PERCENT)


    # Does the model needs to be saved?
    if input('Save Model? [y/N] ') == "y":
        saveModel = True;


    #Training on a Convolutional Neural Network
    printHeader("Emotional Training with a Convolutional Neural Network")
    cnn.convolutional_neural_network(training_images, test_images, y_train, y_test, img_rows, img_cols, saveModel)
